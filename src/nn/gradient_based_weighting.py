import torch
import torch.nn as nn

class PINNWeighting:
    def __init__(self, model, cfg, device, loss_dimension, wandb_run, beta=0.99):
        self.model = model
        self.weights = cfg.nn.weighting.weights # Weights will be initialized based on number of residuals dynamically
        self.scheme = cfg.nn.weighting.update_weight_method
        self.update_weights_freq = cfg.nn.weighting.update_weights_freq
        self.beta = beta
        self.lambda_dn = None 
        self.bias_correction = cfg.nn.weighting.get("bias_correction", False)
        self.dn_counter = 0
        self.wandb_run = wandb_run
        self.loss_dimension = (1 if (cfg.nn.weighting.flag_mean_weights or cfg.nn.time_factored_loss) else loss_dimension)
        self.device = device
        self.initialize_weights()
        self.epoch_flag = -1
        self.smooth = {
            "data": None,
            "dt": None,
            "pinn": None,
            "ic": None,
        }

    def initialize_weights(self):
        # check that the weights are valid and positive
        if self.weights is not None:
            if len(self.weights) != 4:
                raise ValueError("Weights must be a list of 4 values: [loss_data, loss_dt, loss_pinn, loss_pinn_ic]")
            if any(weight < 0 for weight in self.weights):
                raise ValueError("Weights must be positive")
        else:
            raise ValueError("Correct Weights must be provided")
        print("Weights initialized as: ", self.weights," are updated with scheme: ", self.scheme, ("every " + str(self.update_weights_freq) + " epochs") if self.scheme!="Static" else "")
        # Initialize weights if not already initialized
        data_term = 1
        col_term = 0.1
        if self.weights is None:
            total_losses = 1 + self.loss_dimension+ self.loss_dimension + 1 # 1 for loss_data, 1 for loss_pinn_ic and self.loss_dimension for each loss_dt and loss_pinn
            self.weights = torch.ones(total_losses, device=self.device, dtype=torch.float32, requires_grad=(self.scheme == 'Sam'))
            # i want to have a balancing factor for the loss terms with 1, 1,0.1,0.1 for loss_data, loss_dt, loss_pinn, loss_pinn_ic
            self.balancing_term = torch.tensor([1,0.01,0.01,0.01], device=self.device, dtype=torch.float32, requires_grad=False)
        else:
            # Update weights using the scheme and the count of loss components
            updated_weights = (
                [self.weights[0]] +  # For loss_data
                [self.weights[1]] * (self.loss_dimension) +  # For each term in loss_dt
                [self.weights[2]] * (self.loss_dimension) +  # For each term in loss_pinn
                [self.weights[3]]  # For loss_pinn_ic
            )
            self.weights = torch.tensor(updated_weights, device=self.device, dtype=torch.float32, requires_grad=(self.scheme == 'Sam'))
            # i want to have a balancing factor for the loss terms with 1, 1,0.1,0.1 for loss_data, loss_dt, loss_pinn, loss_pinn_ic, maybe i have multiple weights thoug...
            
            balancing_term = ([1] + [0.01] * (self.loss_dimension) + [0.01] * (self.loss_dimension) + [0.01])
            self.balancing_term = torch.tensor(balancing_term, device=self.device, dtype=torch.float32, requires_grad=False)

        # Initialize weight mask
        self.weight_mask = torch.where(self.weights == 0,torch.tensor(0.0, device=self.weights.device),torch.tensor(1.0, device=self.weights.device))

        # If using Sam scheme, ensure weights are parameters
        if self.scheme == 'Sam':
            self.soft_adaptive_weights = nn.Parameter(self.weights)
            self.weights = self.soft_adaptive_weights
        self.log_weights(epoch = 0)
        return


    def compute_weighted_loss(self, loss_data, loss_dt, loss_pinn, loss_pinn_ic,epoch):#, iteration_count=None
        # Calculate losses
        # If loss_dimension is 1, then we need to take mean of the losses
        if self.loss_dimension == 1:
            loss_dt = torch.mean(torch.stack(loss_dt))
            loss_pinn = torch.mean(torch.stack(loss_pinn))

        #SOS multiply with loss_dimension to balance it
        loss_data_ = self.weights[0]  * loss_data * self.loss_dimension
        loss_dt_ = [self.weights[1+i] * (loss_dt[i] if self.loss_dimension > 1 else loss_dt) for i in range(self.loss_dimension)]
        loss_pinn_ = [self.weights[1+self.loss_dimension+i] * (loss_pinn[i] if self.loss_dimension > 1 else loss_pinn) for i in range(self.loss_dimension)]
        loss_pinn_ic_ = self.weights[-1] * loss_pinn_ic * self.loss_dimension
        individual_weighted_losses = [loss_data_] + loss_dt_ + loss_pinn_ + [loss_pinn_ic_]
        #individual_weighted_losses = [torch.tensor(0.0, device=self.device) if torch.isnan(loss) else loss for loss in individual_weighted_losses]
        
        # Calculate total loss
        self.total_loss = torch.stack(individual_weighted_losses).sum()        
        # Update weights with the desired frequency, iteration_count is used to check if the update is needed and not epoch due to lbfgs internal iterations
        if (epoch + 1) % self.update_weights_freq == 0 and self.scheme not in ['Sam', 'MA', 'ID', 'DN']: 
            if epoch != self.epoch_flag: # To avoid multiple updates in the same epoch
                self.update_weights(individual_weighted_losses, epoch)
                #print("Updated Weights", self.weights.tolist())
                self.epoch_flag = epoch
        return self.total_loss, individual_weighted_losses


    def update_weights(self, losses, epoch):
        if self.scheme == 'Gradient':
            # Gradient-based weighting update
            grad_norms = []
            for loss in losses:
                #check if weight_mask is 0 or not
                if self.weight_mask[len(grad_norms)] == 0:
                    grad_norms.append(0)
                    continue
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                grad_norm = torch.norm(torch.stack([torch.norm(p.grad) for p in self.model.parameters() if p.grad is not None]))
                # Check if grad_norm is NaN and skip if it is
                if torch.isnan(grad_norm):
                    print("Warning: NaN gradient norm detected.")
                    grad_norms.append(torch.tensor(0.0)) # Append 0 to avoid division by zero
                else:
                    grad_norms.append(grad_norm.item())
                
            grad_norms = torch.tensor(grad_norms)
            grad_norms = torch.nan_to_num(grad_norms, nan=0.0)  # Replace NaN with 0

            grad_norms_avg = torch.mean(grad_norms)
            new_weights = grad_norms_avg / (grad_norms + 1e-8)
            #print('grad norms are', grad_norms)
            #print('grad norms avg is', grad_norms_avg)
            #print('New weights are', new_weights)

        elif self.scheme == 'Ntk':
            # NTK-based weighting update
            ntk_traces = []
            for loss in losses:
                if self.weight_mask[len(ntk_traces)] == 0:
                    ntk_traces.append(0)
                    continue
                
                self.model.zero_grad()
                loss.backward(retain_graph=True)
                ntk_trace = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        ntk_trace += torch.sum(param.grad ** 2)
                if torch.isnan(ntk_trace):
                    print("Warning: NaN NTK trace detected.")
                    ntk_traces.append(torch.tensor(0.0)) 
                else:
                    ntk_traces.append(ntk_trace.item())
            
            ntk_traces = torch.tensor(ntk_traces, device=self.device)
            ntk_traces = torch.nan_to_num(ntk_traces, nan=0.0)  # Replace NaN with 0
            ntk_traces_avg = torch.mean(ntk_traces)
            # To avoid division by zero, add a small epsilon
            new_weights = ntk_traces_avg / (ntk_traces + 1e-8)
            #print('ntk norms are', ntk_traces)
            #print('ntk norms avg is', ntk_traces_avg)
            #print('New weights are', new_weights)

        elif self.scheme == 'Sam':
            # SAM-based weighting update
            epsilon = 1e-8
            
            with torch.no_grad():
                self.soft_adaptive_weights += self.weight_mask * 0.1 / (self.soft_adaptive_weights.grad + epsilon)
                self.log_weights(epoch)
                print("0.1* sam",  0.1 / (self.soft_adaptive_weights.grad + epsilon))
                self.soft_adaptive_weights.grad.zero_()
                
                self.weights = self.soft_adaptive_weights 
                print(self.weights)
                
            return
            
        elif self.scheme == 'Static':
            return
        
        elif self.scheme == "DN":
            # Deguchi (2023) dynamic normalisation for group-level losses.
            # losses: [data, dt_mean, pinn_mean, ic]
            eps = 1e-12
            grad_norms = []

            # Compute gradient norms for each group loss
            for l in losses:
                self.model.zero_grad()
                l.backward(retain_graph=True)
                g = torch.norm(
                    torch.stack([
                        p.grad.detach().flatten().norm()
                        for p in self.model.parameters()
                        if p.grad is not None
                    ])
                )
                grad_norms.append(g + eps)

            grad_norms = torch.stack(grad_norms).to(self.device)  # shape: [4]

            # Index of physics / PDE loss in [data, dt, pinn, ic]
            idx_pde = 2
            grad_pde = grad_norms[idx_pde]

            # Eq. (22): λ̂_j = ||∇L_pde|| / ||∇L_j||
            lambda_hat = grad_pde / grad_norms

            # Normalise λ̂ so its mean is 1 (Eq. 24–25 style)
            lambda_hat = lambda_hat / lambda_hat.mean()

            # Initialisation or EMA update (Eq. 23)
            if self.lambda_dn is None:
                self.lambda_dn = lambda_hat.detach()
                self.dn_counter = 0
            else:
                b = self.beta
                self.lambda_dn = b * self.lambda_dn + (1.0 - b) * lambda_hat

            # Bias correction (Eq. 26)
            if self.bias_correction:
                self.dn_counter += 1
                correction = 1.0 - (self.beta ** self.dn_counter)
                new_weights = self.lambda_dn / correction
            else:
                new_weights = self.lambda_dn

            # Optional final re-normalization: keep mean ~1
            new_weights = new_weights / new_weights.mean()

            # Apply balancing term and mask (both length 4)
            new_weights = new_weights * self.balancing_term.to(self.device)
            new_weights = new_weights * self.weight_mask

            # Assign back safely
            with torch.no_grad():
                self.weights = new_weights.detach()

            print("[DN-PINN] Updated group weights:", self.weights.tolist())
            self.log_weights(epoch)
            return

        else:
            raise ValueError("Unknown weighting scheme. Choose either 'gradient' or 'ntk' or 'sam'.")
        
        # Update weights using moving average
        self.weights = ( self.weights * self.weight_mask + self.balancing_term *  new_weights.to(self.weights.device)) * self.weight_mask
        print("new weights 0.01*",new_weights)
        print(self.weights)
        
        self.log_weights(epoch)

        return
    
    def log_weights(self, epoch):
        if self.scheme == 'Sam':
            if self.wandb_run is not None and epoch!=0:
                log_data = {f"weights {i}": self.soft_adaptive_weights[i] for i in range(len(self.soft_adaptive_weights))}
                log_data.update({f"soft_adaptive_weights grad{i}": self.soft_adaptive_weights.grad[i] for i in range(len(self.soft_adaptive_weights))})
                log_data["epoch"] = epoch
                self.wandb_run.log(log_data)

        else:
            if self.wandb_run is not None:
                for i in range(len(self.weights)):
                    self.wandb_run.log({"weights"+str(i): self.weights[i], 'epoch': epoch})
        return
    
    def log_losses(self, loss, epoch, name):

        if self.wandb_run is not None:
            for i in range(len(loss)):
                self.wandb_run.log({name[i]: loss[i], 'epoch': epoch})
        return
    
    def update_smoothing(self, L_data, L_dt, L_pinn, L_ic):
        """
        Update exponential moving averages of each loss term.
        ALWAYS reduce per-state losses to a scalar.
        """

        #  Ensure dt and pinn become scalars 
        if torch.is_tensor(L_dt):
            if L_dt.ndim > 0:
                L_dt = L_dt.mean()
        elif isinstance(L_dt, list):
            L_dt = torch.stack(L_dt).mean()

        if torch.is_tensor(L_pinn):
            if L_pinn.ndim > 0:
                L_pinn = L_pinn.mean()
        elif isinstance(L_pinn, list):
            L_pinn = torch.stack(L_pinn).mean()

        # Ensure L_data and L_ic are scalars too
        if torch.is_tensor(L_data) and L_data.ndim > 0:
            L_data = L_data.mean()

        if torch.is_tensor(L_ic) and L_ic.ndim > 0:
            L_ic = L_ic.mean()

        #  Initialize smoothing 
        if self.smooth["data"] is None:
            print("Initializing smoothing")
            self.smooth["data"] = L_data
            self.smooth["dt"]   = L_dt
            self.smooth["pinn"] = L_pinn
            self.smooth["ic"]   = L_ic
            return

        #  Update EMA 
        b = self.beta
        self.smooth["data"] = b * self.smooth["data"] + (1 - b) * L_data
        self.smooth["dt"]   = b * self.smooth["dt"]   + (1 - b) * L_dt
        self.smooth["pinn"] = b * self.smooth["pinn"] + (1 - b) * L_pinn
        self.smooth["ic"]   = b * self.smooth["ic"]   + (1 - b) * L_ic

        #  Debug print 
        print(f"[SMOOTH] data={float(self.smooth['data']):.4e}, "
            f"dt={float(self.smooth['dt']):.4e}, "
            f"pinn={float(self.smooth['pinn']):.4e}, "
            f"ic={float(self.smooth['ic']):.4e}")
    
    def update_weights_MA(self, epoch):
        """
        Max-Average (MA) adaptive weighting.
        Uses exponentially-smoothed losses stored in self.smooth.
        Expands 4 group-level weights → full expanded weight vector.
        Includes detailed debug prints for analysis.
        """

        # Verify smoothing is initialized
        if (
            self.smooth["data"] is None or
            self.smooth["dt"]   is None or
            self.smooth["pinn"] is None or
            self.smooth["ic"]   is None
        ):
            print("MA skipped — smoothing not initialized yet.")
            return

        # Gather smoothed group losses
        L_data = float(self.smooth["data"])
        L_dt   = float(self.smooth["dt"])
        L_pinn = float(self.smooth["pinn"])
        L_ic   = float(self.smooth["ic"])

        print(f"Smoothed losses → data={L_data:.4e}, dt={L_dt:.4e}, pinn={L_pinn:.4e}, ic={L_ic:.4e}")

        L_vec = torch.tensor([L_data, L_dt, L_pinn, L_ic], dtype=torch.float32, device=self.device)
        eps = 1e-8

        # Compute MA weights
        L_avg = torch.mean(L_vec)
        print(f"Loss average (L_avg): {L_avg.item():.4e}")

        w_raw = L_avg / (L_vec + eps)
        print(f"Raw MA weights (before norm): {w_raw.tolist()}")

        # Normalize so weights sum to 1 (optional but stabilizing)
        w_norm = w_raw / torch.sum(w_raw)
        print(f"Normalized MA weights: {w_norm.tolist()}")

        # Expand MA group weights to full vector shape
        w_data = w_norm[0]
        w_dt   = w_norm[1]
        w_pinn = w_norm[2]
        w_ic   = w_norm[3]

        full_weights = (
            [w_data] +
            [w_dt]   * self.loss_dimension +
            [w_pinn] * self.loss_dimension +
            [w_ic]
        )

        full_weights = torch.tensor(full_weights, dtype=torch.float32, device=self.device)

        # Apply mask (freeze zero-weights)
        full_weights = full_weights * self.weight_mask

        # Assign back safely
        with torch.no_grad():
            self.weights = full_weights.detach()
