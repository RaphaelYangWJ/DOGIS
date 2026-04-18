import torch
import torch.nn as nn



# Class Flow Matching
class FlowMatching(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model # backbone

    def get_xt(self, x1, x0, t):
        t_view = t.view(-1, *([1] * (x1.dim() - 1)))
        xt = (1 - t_view) * x0 + t_view * x1
        return xt

    def get_velocity_target(self, x1, x0):
        return x1 - x0

    # Forward
    def forward(self, x1, spatial_feat, global_feat, mask, op_weight, cfg):
        b = x1.shape[0]
        device = x1.device

        # 1. Sampling t ~ U(0, 1)
        t = torch.rand(b, device=device)

        # 2. Sampling noise x0 ~ N(0, I)
        x0 = torch.randn_like(x1)

        # 3. Generate interpolated samples xt and target velocity ut
        xt = self.get_xt(x1, x0, t)
        ut = self.get_velocity_target(x1, x0)

        # 4. CFG
        if cfg is not False:
            mask = torch.rand(b, device=device) < cfg
            if global_feat is not None:
                mask_feat = mask.view(-1, 1).expand_as(global_feat)
                global_feat = torch.where(mask_feat, torch.zeros_like(global_feat), global_feat)
            if spatial_feat is not None:
                mask_obs = mask.view(-1, 1, 1, 1).expand_as(spatial_feat)
                spatial_feat = torch.where(mask_obs, torch.zeros_like(spatial_feat), spatial_feat)

        
        # 5. Vt generation
        v_pred = self.model(xt, t, spatial_feat, global_feat)

        # 6. Project onto t = 1 (real data) to obtain the denoised result x_hat
        # Expand t to a broadcastable dimension [B, 1, 1, 1]
        t_expand = t.view(b, 1, 1, 1)

        # By traveling along the predicted velocity direction for the remaining time (1 - t), a clean physics prediction can be obtained.
        x_hat = xt + (1.0 - t_expand) * v_pred

        # 7. Calculate the Time-dependent Physics Weight
        # When t is close to 0 (pure noise), x_hat predictions are extremely inaccurate, and calculating the physics loss at this point will produce a toxic gradient.
        # We use a power function of t as dynamic weights, allowing the FNO loss to only strongly intervene in the later stages of denoising (t > 0.5).

        # op_weight is the base physical weight you pass in from outside (e.g., 0.1)
        # t_expand**3 makes the weight only 0.001 when t=0.1, and the weight reaches 0.729 when t=0.9.
        adaptive_physics_weight = op_weight * (t_expand ** 3) 

        return v_pred, ut, x_hat, adaptive_physics_weight


    # Standard Sampler
    @torch.no_grad()
    def sample(self, field_shape, obs_shape, steps=50, device="cuda", spatial_feat=None, global_feat=None):
        x_inv = torch.randn(field_shape, device=device)

        dt = 1.0 / steps

        for i in range(steps):
            t_val = i / steps
            t = torch.full((field_shape[0],), t_val, device=device)
            v_pred = self.model(x_inv, t, spatial_feat, global_feat)
            x_inv = x_inv + v_pred * dt

        # Return outputs
        return x_inv


    def FNO_sampler(self, field_shape, steps=50, device="cuda", 
               spatial_feat=None, global_feat=None, 
               cfg_scale=1.0,
               use_physics_guidance=False,
               fno_model=None,
               y_obs_sparse=None,
               mask=None,
               guidance_scale=10.0,
               guidance_start_t=0.5):
        
        # 1. Starting with pure noise x_t (assuming t=0 is noise)
        x_t = torch.randn(field_shape, device=device)
        dt = 1.0 / steps
        b = field_shape[0]

        for i in range(steps):
            t_val = i / steps
            t = torch.full((b,), t_val, device=device)
            t_expand = t.view(b, 1, 1, 1)

            if use_physics_guidance and t_val >= guidance_start_t:
                with torch.enable_grad():
                    # The computation graph must be disconnected from the current state, and the requirement for gradients must be declared.
                    x_t_in = x_t.detach().requires_grad_(True)
                    
                    # 1. Predict flow field velocity v_pred
                    v_pred = self.model(x_t_in, t, spatial_feat, global_feat)
                    
                    # 2. Linear extrapolation, projected onto the target (t=1, clean physics field)
                    x_hat = x_t_in + (1.0 - t_expand) * v_pred
                    
                    # 3. FNO forward modeling and calculation of physical loss
                    dense_obs_pred = fno_model(x_hat)
                    diff = (dense_obs_pred - y_obs_sparse) * mask.unsqueeze(1)
                    # You can use either the mean or the summation; here, we'll use the summation. The gradient magnitude is controlled by guidance_scale.
                    loss_physics = torch.sum(diff ** 2) 
                    
                    # 4. Calculate the physical gradient of the input x_t_in
                    grad_x = torch.autograd.grad(loss_physics, x_t_in)[0]

                    # 5. Correcting the Velocity Field (Test-Time Guidance)
                    # Introduce a time decay coefficient. As t approaches 1, (1-t) decreases. Here, t_val is used to increase the weight.
                    dynamic_gamma = guidance_scale * (t_val ** 2) 
                    
                    # Correct the speed in the direction that reduces the error
                    v_guided = v_pred.detach() - dynamic_gamma * grad_x
                    
                    # 6. Euler Update (Note: Use detach to prevent the computation graph from accumulating and overflowing memory)
                    x_t = x_t.detach() + v_guided * dt

            else:
                with torch.no_grad():
                    v_pred = self.model(x_t, t, spatial_feat, global_feat)

                    x_t = x_t + v_pred * dt

        return x_t.detach() # 输出最终的清晰物理场

