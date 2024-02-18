import torch
import torch.nn.functional as F

class UnbatchedParticleFilter:
    def __init__(self, resampling_method, device):
        resampling_fns = {"multinomial": self.multinomial_resampling, "systematic": self.systematic_resampling,
                          "stratified": self.stratified_resampling}

        self._resample = resampling_fns[resampling_method]
        self.device = device

    @staticmethod
    def multinomial_resampling(nu):
        a = nu.multinomial(len(nu), replacement=True)
        return a

    def resample(self, nu):
        assert len(nu.shape) == 1
        return self._resample(nu)

    def _sampling_help(self, offset, nu):
        num_particles = len(nu)
        base = (1 / num_particles) * torch.arange(num_particles, device=self.device)
        p = base + offset
        p = p.unsqueeze(1)
        nu_cumsum = nu.cumsum(dim=0).unsqueeze(0)
        indices = num_particles - torch.sum(p < nu_cumsum, dim=-1)
        return indices

    def systematic_resampling(self, nu):
        num_particles = len(nu)
        offset = 1 / num_particles * torch.rand(1, device=self.device)
        return self._sampling_help(offset, nu)

    def stratified_resampling(self, nu):
        num_particles = len(nu)
        offset = (1 / num_particles) * torch.rand(num_particles, device=self.device)
        return self._sampling_help(offset, nu)

    @staticmethod
    def compute_ess(w):
        assert len(w.shape) == 1
        n_eff = 1 / torch.sum(w ** 2, dim=-1)
        return n_eff

    def normalize(self, vec):
        assert len(vec.shape) == 1
        return F.normalize(vec, p=1, dim=-1)


class BatchedParticleFilter:
    def __init__(self, num_particles, resampling_method, device):
        self.device = device
        self.num_particles = num_particles
        resampling_fns = {"multinomial": self.multinomial_resampling, "systematic": self.systematic_resampling,
                          "stratified": self.stratified_resampling}

        self._resample = resampling_fns[resampling_method]
        self.device = device

    def resample(self, nu):
        assert len(nu.shape) == 2
        return self._resample(nu)

    def multinomial_resampling(self, nu):
        a = nu.multinomial(self.num_particles, replacement=True).flatten()
        particle_offset = (self.num_particles * torch.arange(nu.shape[0],
                                                             device=self.device)).repeat_interleave(self.num_particles)
        a = a + particle_offset
        return a.flatten()

    def _sampling_help(self, p, nu):
        nu_cumsum = nu.cumsum(dim=-1)
        p = p.flatten().unsqueeze(-1)
        nu_cumsum = nu_cumsum.repeat_interleave(self.num_particles, dim=0)
        indices = self.num_particles - torch.sum(p < nu_cumsum, dim=-1)
        indices = indices + (self.num_particles * torch.arange(nu.shape[0], 
                                                             device=self.device)).repeat_interleave(self.num_particles)
        return indices.flatten()

    def systematic_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], 1), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    def stratified_resampling(self, nu):
        base = (1 / self.num_particles) * torch.arange(self.num_particles, device=self.device).reshape((1, self.num_particles))
        offset = (1 / self.num_particles) * torch.rand((nu.shape[0], self.num_particles), device=self.device)
        p = base + offset
        return self._sampling_help(p, nu)

    @staticmethod
    def compute_ess(w):
        assert len(w.shape) == 2
        n_eff = 1 / torch.sum(w ** 2, dim=-1)
        return n_eff.flatten()

    def normalize(self, vec):
        assert len(vec.shape) == 1
        vec = vec.reshape((-1, self.num_particles))
        return F.normalize(vec, p=1, dim=-1).flatten()
