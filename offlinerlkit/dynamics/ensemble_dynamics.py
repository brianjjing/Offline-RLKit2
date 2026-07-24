import os
import numpy as np
import torch
import torch.nn as nn

from typing import Callable, List, Tuple, Dict, Optional
from offlinerlkit.dynamics import BaseDynamics
from offlinerlkit.utils.scaler import StandardScaler
from offlinerlkit.utils.logger import Logger


class EnsembleDynamics(BaseDynamics):
    def __init__(
        self,
        model: nn.Module,
        optim: torch.optim.Optimizer,
        scaler: StandardScaler,
        terminal_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        penalty_coef: float = 0.0,
        uncertainty_mode: str = "aleatoric",
        classifier=None, #Added for DBG
        penalty_type="linear" #Added for DBG
    ) -> None:
        super().__init__(model, optim)
        self.scaler = scaler
        self.terminal_fn = terminal_fn
        self._penalty_coef = penalty_coef
        self._uncertainty_mode = uncertainty_mode
        self.classifier_name = classifier['name'] if 'name' in classifier else None   # Added for DBG
        self.classifier_mean = classifier['mean'] if 'mean' in classifier else None   # Added for DBG
        self.classifier_std  = classifier['std']  if 'std'  in classifier else None   # Added for DBG
        self.classifier_model = classifier['model'] #Added for DBG
        self.classifier_thr = classifier['thr'] #Added for DBG
        self.device = self.model.device #Added fror DBG
        self.type = penalty_type #Added for DBG


    def _return_kde_penalty(self, state, action, reward=None, type= "linear", alpha = 0.1):
        
        """
        Version with optimized IQR filtering.
        """ 
        
        if self.classifier_name is None:
            input_np = np.concatenate([state, action], axis=1)
        else:
            input_np = torch.cat([action.squeeze(1), reward.view(-1, 1)], dim=1)
        log_probs = self.classifier_model.score_samples(input_np, self.device)
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().cpu().numpy()
        # print('log_probs mean and std: ', log_probs.mean(), log_probs.std())
        if self.classifier_mean is not None and self.classifier_std is not None:
            log_probs = (log_probs - self.classifier_mean) / self.classifier_std
        if type == "linear":
          
            log_weight = self.classifier_thr - log_probs #high means more likely to be OOD
            q1, q3 = np.percentile(log_weight, [25, 75])
            upper_bound = q3 + 1.5 * (q3 - q1)
            weight = np.clip(log_weight, a_min=0, a_max=upper_bound)
        elif type == "inverse":
            weight = np.where(
                log_probs < self.classifier_thr,
                np.exp(self.classifier_thr) / (np.exp(log_probs) + 1e-6),
                0.0
            )
        elif type == "tanh":
            log_weight = (np.tanh(0.1*(-log_probs + self.classifier_thr)))
            weight = np.clip(log_weight, a_min=0, a_max=None)
        elif type == "tanh_reg":
            weight = (np.tanh(0.1*(-log_probs + self.classifier_thr)))
            # print(weight.mean(), weight.std())
        elif type == "softplus": #smooth and stable
            weight = np.log(1 + np.exp(-log_probs)).numpy()
        #plot the weights in histogram
        # Plotting moved to algo/mopo.py:rollout_transitions() for better frequency control

        return weight


    @ torch.no_grad()
    def step(
        self,
        obs: np.ndarray,
        action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        "imagine single forward step"
        obs_act = np.concatenate([obs, action], axis=-1)
        obs_act = self.scaler.transform(obs_act)
        mean, logvar = self.model(obs_act)
        mean = mean.cpu().numpy()
        logvar = logvar.cpu().numpy()
        mean[..., :-1] += obs
        std = np.sqrt(np.exp(logvar))

        ensemble_samples = (mean + np.random.normal(size=mean.shape) * std).astype(np.float32)

        # choose one model from ensemble
        num_models, batch_size, _ = ensemble_samples.shape
        model_idxs = self.model.random_elite_idxs(batch_size)
        samples = ensemble_samples[model_idxs, np.arange(batch_size)]
        
        next_obs = samples[..., :-1]
        reward = samples[..., -1:]
        terminal = self.terminal_fn(obs, action, next_obs)
        info = {}
        info["raw_reward"] = reward

        # DBG PENALTY REWARDS:
        act = action #For compliance w the DBG block

        penalty_coeff = self._penalty_coef
        penalty_learned_var = True
        if penalty_coeff != 0:
            if not penalty_learned_var:
                ensemble_means_obs = pred_diff_means[:, :, :-1]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)  # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0, :obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)  # distance in obs space
                penalty = np.max(dists, axis=0)  # max distances over models
            else:
                # penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)
                penalty = self._return_kde_penalty(next_obs, act, type= self.type)
            
            penalty = np.expand_dims(penalty, 1)  # Matching shape w/ reward jst as the original implementation did.
            assert penalty.shape == reward.shape

            penalized_rewards = reward - penalty_coeff * penalty
            info = {'penalty': penalty, 'penalized_rewards': penalized_rewards}

        else:
            penalized_rewards = reward
            info = {'penalized_rewards': penalized_rewards}

        # END DBG PENALTY REWARDS


        #UNCOMMENT BELOW FOR ORIGINAL PENALTY REWARDS:
        
        # if self._penalty_coef:
        #     if self._uncertainty_mode == "aleatoric":
        #         penalty = np.amax(np.linalg.norm(std, axis=2), axis=0)
        #     elif self._uncertainty_mode == "pairwise-diff":
        #         next_obses_mean = mean[..., :-1]
        #         next_obs_mean = np.mean(next_obses_mean, axis=0)
        #         diff = next_obses_mean - next_obs_mean
        #         penalty = np.amax(np.linalg.norm(diff, axis=2), axis=0)
        #     elif self._uncertainty_mode == "ensemble_std":
        #         next_obses_mean = mean[..., :-1]
        #         penalty = np.sqrt(next_obses_mean.var(0).mean(1))
        #     else:
        #         raise ValueError
        #     penalty = np.expand_dims(penalty, 1).astype(np.float32)
        #     assert penalty.shape == reward.shape
        #     reward = reward - self._penalty_coef * penalty
        #     info["penalty"] = penalty
        
        #UNCOMMENT ABOVE FOR ORIGINAL PENALTY REWARDS  

        return next_obs, penalized_rewards, terminal, info
    
    @ torch.no_grad()
    def sample_next_obss(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        num_samples: int
    ) -> torch.Tensor:
        obs_act = torch.cat([obs, action], dim=-1)
        obs_act = self.scaler.transform_tensor(obs_act)
        mean, logvar = self.model(obs_act)
        mean[..., :-1] += obs
        std = torch.sqrt(torch.exp(logvar))

        mean = mean[self.model.elites.data.cpu().numpy()]
        std = std[self.model.elites.data.cpu().numpy()]

        samples = torch.stack([mean + torch.randn_like(std) * std for i in range(num_samples)], 0)
        next_obss = samples[..., :-1]
        return next_obss

    def format_samples_for_training(self, data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        obss = data["observations"]
        actions = data["actions"]
        next_obss = data["next_observations"]
        rewards = data["rewards"]
        delta_obss = next_obss - obss
        inputs = np.concatenate((obss, actions), axis=-1)
        targets = np.concatenate((delta_obss, rewards), axis=-1)
        return inputs, targets

    def train(
        self,
        data: Dict,
        logger: Logger,
        max_epochs: Optional[float] = None,
        max_epochs_since_update: int = 5,
        batch_size: int = 256,
        holdout_ratio: float = 0.2,
        logvar_loss_coef: float = 0.01
    ) -> None:
        inputs, targets = self.format_samples_for_training(data)
        data_size = inputs.shape[0]
        holdout_size = min(int(data_size * holdout_ratio), 1000)
        train_size = data_size - holdout_size
        train_splits, holdout_splits = torch.utils.data.random_split(range(data_size), (train_size, holdout_size))
        train_inputs, train_targets = inputs[train_splits.indices], targets[train_splits.indices]
        holdout_inputs, holdout_targets = inputs[holdout_splits.indices], targets[holdout_splits.indices]

        self.scaler.fit(train_inputs)
        train_inputs = self.scaler.transform(train_inputs)
        holdout_inputs = self.scaler.transform(holdout_inputs)
        holdout_losses = [1e10 for i in range(self.model.num_ensemble)]

        data_idxes = np.random.randint(train_size, size=[self.model.num_ensemble, train_size])
        def shuffle_rows(arr):
            idxes = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxes]

        epoch = 0
        cnt = 0
        logger.log("Training dynamics:")
        while True:
            epoch += 1
            train_loss = self.learn(train_inputs[data_idxes], train_targets[data_idxes], batch_size, logvar_loss_coef)
            new_holdout_losses = self.validate(holdout_inputs, holdout_targets)
            holdout_loss = (np.sort(new_holdout_losses)[:self.model.num_elites]).mean()
            logger.logkv("loss/dynamics_train_loss", train_loss)
            logger.logkv("loss/dynamics_holdout_loss", holdout_loss)
            logger.set_timestep(epoch)
            logger.dumpkvs(exclude=["policy_training_progress"])

            # shuffle data for each base learner
            data_idxes = shuffle_rows(data_idxes)

            indexes = []
            for i, new_loss, old_loss in zip(range(len(holdout_losses)), new_holdout_losses, holdout_losses):
                improvement = (old_loss - new_loss) / old_loss
                if improvement > 0.01:
                    indexes.append(i)
                    holdout_losses[i] = new_loss
            
            if len(indexes) > 0:
                self.model.update_save(indexes)
                cnt = 0
            else:
                cnt += 1
            
            if (cnt >= max_epochs_since_update) or (max_epochs and (epoch >= max_epochs)):
                break

        indexes = self.select_elites(holdout_losses)
        self.model.set_elites(indexes)
        self.model.load_save()
        self.save(logger.model_dir)
        self.model.eval()
        logger.log("elites:{} , holdout loss: {}".format(indexes, (np.sort(holdout_losses)[:self.model.num_elites]).mean()))
    
    def learn(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int = 256,
        logvar_loss_coef: float = 0.01
    ) -> float:
        self.model.train()
        train_size = inputs.shape[1]
        losses = []

        for batch_num in range(int(np.ceil(train_size / batch_size))):
            inputs_batch = inputs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = targets[:, batch_num * batch_size:(batch_num + 1) * batch_size]
            targets_batch = torch.as_tensor(targets_batch).to(self.model.device)
            
            mean, logvar = self.model(inputs_batch)
            inv_var = torch.exp(-logvar)
            # Average over batch and dim, sum over ensembles.
            mse_loss_inv = (torch.pow(mean - targets_batch, 2) * inv_var).mean(dim=(1, 2))
            var_loss = logvar.mean(dim=(1, 2))
            loss = mse_loss_inv.sum() + var_loss.sum()
            loss = loss + self.model.get_decay_loss()
            loss = loss + logvar_loss_coef * self.model.max_logvar.sum() - logvar_loss_coef * self.model.min_logvar.sum()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            losses.append(loss.item())
        return np.mean(losses)
    
    @ torch.no_grad()
    def validate(self, inputs: np.ndarray, targets: np.ndarray) -> List[float]:
        self.model.eval()
        targets = torch.as_tensor(targets).to(self.model.device)
        mean, _ = self.model(inputs)
        loss = ((mean - targets) ** 2).mean(dim=(1, 2))
        val_loss = list(loss.cpu().numpy())
        return val_loss
    
    def select_elites(self, metrics: List) -> List[int]:
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        elites = [pairs[i][1] for i in range(self.model.num_elites)]
        return elites

    def save(self, save_path: str) -> None:
        torch.save(self.model.state_dict(), os.path.join(save_path, "dynamics.pth"))
        self.scaler.save_scaler(save_path)
    
    def load(self, load_path: str) -> None:
        self.model.load_state_dict(torch.load(os.path.join(load_path, "dynamics.pth"), map_location=self.model.device))
        self.scaler.load_scaler(load_path)
