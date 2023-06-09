import os
import os.path as osp
import torch
import torch.nn.functional as F
import numpy as np
import itertools
from tensorboardX import SummaryWriter

from NN.losses import make_criteria
from utils.base import logger

class GPS:
    def __init__(self,
                 init_mode: str = 'random_synthesis',
                 noise_sigma: float = 1.0,
                 coarse_ratio: float = 0.2,
                 coarse_ratio_factor: float = 6,
                 pyr_factor: float = 0.75,
                 num_stages_limit: int = -1,
                 device: str = 'cuda:0',
                 silent: bool = False
                 ):
        '''
        Args:
            init_mode:
                - 'random_synthesis': init with random seed
                - 'random': init with random seed
            noise_sigma: float = 1.0, random noise.
            coarse_ratio: float = 0.2, ratio at the coarse level.
            pyr_factor: float = 0.75, pyramid factor.
            num_stages_limit: int = -1, no limit.
            device: str = 'cuda:0', default device.
            silent: bool = False, mute the output.
        '''
        self.init_mode = init_mode
        self.noise_sigma = noise_sigma
        self.coarse_ratio = coarse_ratio
        self.coarse_ratio_factor = coarse_ratio_factor
        self.pyr_factor = pyr_factor
        self.num_stages_limit = num_stages_limit
        self.device = torch.device(device)
        self.silent = silent

    def _get_pyramid_lengths(self, dest, ext=None):
        """Get a list of pyramid lengths"""
        if self.coarse_ratio == -1:
            self.coarse_ratio = np.around(ext['criteria']['patch_size'] * self.coarse_ratio_factor / dest, 2)

        lengths = [int(np.round(dest * self.coarse_ratio))]
        while lengths[-1] < dest:
            lengths.append(int(np.round(lengths[-1] / self.pyr_factor)))
            if lengths[-1] == lengths[-2]:
                lengths[-1] += 1
        lengths[-1] = dest

        return lengths

    def _get_target_pyramid(self, target, ext=None):
        """Reads a target motion(s) and create a pyraimd out of it. Ordered in increatorch.sing size"""
        self._num_target = len(target)
        lengths = []
        min_len = 10000
        for i in range(len(target)):
            new_length = self._get_pyramid_lengths(len(target[i]), ext)
            min_len = min(min_len, len(new_length))
            if self.num_stages_limit != -1:
                new_length = new_length[:self.num_stages_limit]
            lengths.append(new_length)
        for i in range(len(target)):
            lengths[i] = lengths[i][-min_len:]
        self.pyraimd_lengths = lengths

        target_pyramid = [[] for _ in range(len(lengths[0]))]
        for step in range(len(lengths[0])):
            for i in range(len(target)):
                length = lengths[i][step]
                motion = target[i]
                target_pyramid[step].append(motion.sample(size=length).to(self.device))
                # target_pyramid[step].append(motion.pos2velo(motion.sample(size=length)))
                # motion.motion_data = motion.pos2velo(motion.motion_data)
                # target_pyramid[step].append(motion.sample(size=length))
                # motion.motion_data = motion.velo2pos(motion.motion_data)    

        if not self.silent:
            print('Levels:', lengths)
            for i in range(len(target_pyramid)):
                print(f'Number of clips in target pyramid {i} is {len(target_pyramid[i])}: {[[tgt.min(), tgt.max()] for tgt in target_pyramid[i]]}')

        return target_pyramid

    def _get_initial_motion(self):
        """Prepare the initial motion for optimization"""
        if 'random_synthesis' in str(self.init_mode):
            m = self.init_mode.split('/')[-1]
            if m =='random_synthesis':
                final_length = sum([i[-1] for i in self.pyraimd_lengths])
            elif 'x' in m:
                final_length = int(m.replace('x', '')) * sum([i[-1] for i in self.pyraimd_lengths])
            elif (self.init_mode.split('/')[-1]).isdigit():
                final_length = int(self.init_mode.split('/')[-1])
            else:
                raise ValueError(f'incorrect init_mode: {self.init_mode}')

            self.synthesized_lengths = self._get_pyramid_lengths(final_length)

        else:
            raise ValueError(f'Unsupported init_mode {self.init_mode}')
            
        initial_motion = F.interpolate(torch.cat([self.target_pyramid[0][i] for i in range(self._num_target)], dim=-1),
                                       size=self.synthesized_lengths[0], mode='linear', align_corners=True)
        if self.noise_sigma > 0:
            initial_motion_w_noise = initial_motion + torch.randn_like(initial_motion) * self.noise_sigma
            initial_motion_w_noise = torch.fmod(initial_motion_w_noise, 1.0)
        else:
            initial_motion_w_noise = initial_motion

        if not self.silent:
            print('Synthesized lengths:', self.synthesized_lengths)
            print('Initial motion:', initial_motion.min(), initial_motion.max())
            print('Initial motion with noise:', initial_motion_w_noise.min(), initial_motion_w_noise.max())

        return initial_motion_w_noise

    def run(self, target, mode="backpropagate", ext=None, debug_dir=None):
        '''
        Run the patch-based motion synthesis.

        Args:
            target (torch.Tensor): Target data.
            mode (str): Optimization mode. Support ['backpropagate', 'match_and_blend']
            ext (dict): extra data or constrain.
            debug_dir (str): Debug directory.
        '''
        # preprare data
        self.target_pyramid = self._get_target_pyramid(target, ext)
        self.synthesized = self._get_initial_motion()
        if debug_dir is not None:
            writer = SummaryWriter(log_dir=debug_dir)

        # prepare configuration
        if mode == "backpropagate":
            self.synthesized.requires_grad_(True)
            assert 'criteria' in ext.keys(), 'Please specify a criteria for synthsis.'
            criteria = make_criteria(ext['criteria']).to(self.device)
        elif mode == "match_and_blend":
            self.synthesized.requires_grad_(False)
            assert 'criteria' in ext.keys(), 'Please specify a criteria for synthsis.'
            criteria = make_criteria(ext['criteria']).to(self.device)
        else:
            raise ValueError(f'Unsupported mode: {mode}')

        # perform synthsis
        self.pbar = logger(ext['num_itrs'], len(self.target_pyramid))
        ext['pbar'] = self.pbar
        for lvl, lvl_target in enumerate(self.target_pyramid):
            self.pbar.new_lvl()
            if lvl > 0:
                with torch.no_grad():
                    self.synthesized = F.interpolate(self.synthesized.detach(), size=self.synthesized_lengths[lvl], mode='linear')
                if mode == "backpropagate":
                    self.synthesized.requires_grad_(True)

            if mode == "backpropagate": # direct optimize the synthesized motion
                self.synthesized, losses = GPS.backpropagate(self.synthesized, lvl_target, criteria, ext=ext)
            elif mode == "match_and_blend":
                self.synthesized, losses = GPS.match_and_blend(self.synthesized, lvl_target, criteria, ext=ext)

            criteria.clean_cache()
            if debug_dir:
                for itr in range(len(losses)):
                    writer.add_scalar(f'optimize/losses_lvl{lvl}', losses[itr], itr)
        self.pbar.pbar.close()


        return self.synthesized.detach()

    @staticmethod
    def backpropagate(synthesized, targets, criteria=None, ext=None):
        """
        Minimizes criteria(synthesized, target) for num_steps SGD steps
        Args:
            targets (torch.Tensor): Target data.
            ext (dict): extra configurations.
        """
        if criteria is None:
            assert 'criteria' in ext.keys(), 'Criteria is not set'
            criteria = make_criteria(ext['criteria']).to(synthesized.device)

        optim = None
        if 'optimizer' in ext.keys():
            if ext['optimizer'] == 'Adam':
                optim = torch.optim.Adam([synthesized], lr=ext['lr'])
            elif ext['optimizer'] == 'SGD':
                optim = torch.optim.SGD([synthesized], lr=ext['lr'])
            elif ext['optimizer'] == 'RMSprop':
                optim = torch.optim.RMSprop([synthesized], lr=ext['lr'])
            else:
                print(f'use default RMSprop optimizer')
        optim = torch.optim.RMSprop([synthesized], lr=ext['lr']) if optim is None else optim
        # optim = torch.optim.Adam([synthesized], lr=ext['lr']) if optim is None else optim
        lr_decay = np.exp(np.log(0.333) / ext['num_itrs'])

        # other constraints
        trajectory = ext['trajectory'] if 'trajectory' in ext.keys() else None

        losses = []
        for _i in range(ext['num_itrs']):
            optim.zero_grad()
            
            loss = criteria(synthesized, targets)

            if trajectory is not None: ## velo constrain
                target_traj = F.interpolate(trajectory, size=synthesized.shape[-1], mode='linear')
                # target_traj = F.interpolate(trajectory, size=synthesized.shape[-1], mode='linear', align_corners=False)
                target_velo = ext['pos2velo'](target_traj)
                
                velo_mask = [-3, -1]
                loss += 1 * F.l1_loss(synthesized[:, velo_mask, :], target_velo[:, velo_mask, :])

            loss.backward()
            optim.step()

            # Update staus
            losses.append(loss.item())
            if 'pbar' in ext.keys():
                ext['pbar'].step()
                ext['pbar'].print()

        return synthesized, losses

    @staticmethod
    @torch.no_grad()
    def match_and_blend(synthesized, targets, criteria, ext):
        """
        Minimizes criteria(synthesized, target)
        Args:
            targets (torch.Tensor): Target data.
            ext (dict): extra configurations.
        """
        losses = []
        for _i in range(ext['num_itrs']):
            if 'parts_list' in ext.keys():
                def extract_part_motions(motion, parts_list):
                    part_motions = []
                    n_frames = motion.shape[-1]
                    rot, pos = motion[:, :-3, :].reshape(-1, 6, n_frames), motion[:, -3:, :]

                    for part in parts_list:
                        # part -= 1
                        part = [i -1 for i in part]

                        # print(part)
                        if 0 in part:
                            part_motions += [torch.cat([rot[part].view(1, -1, n_frames), pos.view(1, -1, n_frames)], dim=1)]
                        else:
                            part_motions += [rot[part].view(1, -1, n_frames)]

                    return part_motions
                def combine_part_motions(part_motions, parts_list):
                    assert len(part_motions) == len(parts_list)
                    n_frames = part_motions[0].shape[-1]
                    l = max(list(itertools.chain(*parts_list)))
                    # print(l, n_frames)
                    # motion = torch.zeros((1, (l+1)*6 + 3, n_frames), device=part_motions[0].device)
                    rot = torch.zeros(((l+1), 6, n_frames), device=part_motions[0].device)
                    pos = torch.zeros((1, 3, n_frames), device=part_motions[0].device)
                    div_rot = torch.zeros((l+1), device=part_motions[0].device)
                    div_pos = torch.zeros(1, device=part_motions[0].device)

                    for part_motion, part in zip(part_motions, parts_list):
                        part = [i -1 for i in part]

                        if 0 in part:
                            # print(part_motion.shape)
                            pos += part_motion[:, -3:, :]
                            div_pos += 1
                            rot[part] += part_motion[:, :-3, :].view(-1, 6, n_frames)
                            div_rot[part] += 1
                        else:
                            rot[part] += part_motion.view(-1, 6, n_frames)
                            div_rot[part] += 1
                            
                    # print(div_rot, div_pos)
                    # print(rot.shape)
                    rot = (rot.permute(1, 2, 0) / div_rot).permute(2, 0, 1)
                    pos = pos / div_pos

                    return torch.cat([rot.view(1, -1, n_frames), pos.view(1, 3, n_frames)], dim=1)

                # raw_synthesized = synthesized
                # print(synthesized, synthesized.shape)
                synthesized_part_motions = extract_part_motions(synthesized, ext['parts_list'])
                targets_part_motions = [extract_part_motions(target, ext['parts_list']) for target in targets]

                synthesized = []
                for _j in range(len(synthesized_part_motions)):
                    synthesized_part_motion = synthesized_part_motions[_j]
                    # synthesized += [synthesized_part_motion]
                    targets_part_motion = [target[_j] for target in targets_part_motions]
                    # # print(synthesized_part_motion.shape, targets_part_motion[0].shape)
                    synthesized += [criteria(synthesized_part_motion, targets_part_motion, ext=ext, return_blended_results=True)[0]]

                # print(len(synthesized))
                
                synthesized = combine_part_motions(synthesized, ext['parts_list'])
                # print(synthesized, synthesized.shape)
                # print((raw_synthesized-synthesized > 0.00001).sum())
                # exit()
                # print(synthesized.shape)
                losses = 0

                # exit()
       
            else:
                synthesized, loss = criteria(synthesized, targets, ext=ext, return_blended_results=True)

                # Update staus
                losses.append(loss.item())
                if 'pbar' in ext.keys():
                    ext['pbar'].step()
                    ext['pbar'].print()

        return synthesized, losses

