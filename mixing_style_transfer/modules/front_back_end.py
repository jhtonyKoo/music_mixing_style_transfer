""" Front-end: processing raw data input """
import torch
import torch.nn as nn
import torchaudio.functional as ta_F
import torchaudio



class FrontEnd(nn.Module):
    def __init__(self, channel='stereo', \
                        n_fft=2048, \
                        hop_length=None, \
                        win_length=None, \
                        window="hann", \
                        device=torch.device("cpu")):
        super(FrontEnd, self).__init__()
        self.channel = channel
        self.n_fft = n_fft
        self.hop_length = n_fft//4 if hop_length==None else hop_length
        self.win_length = n_fft if win_length==None else win_length
        if window=="hann":
            self.window = torch.hann_window(window_length=self.win_length, periodic=True).to(device)
        elif window=="hamming":
            self.window = torch.hamming_window(window_length=self.win_length, periodic=True).to(device)


    def forward(self, input, mode):
        # front-end function which channel-wise combines all demanded features
        # input shape : batch x channel x raw waveform
        # output shape : batch x channel x frequency x time

        front_output_list = []
        for cur_mode in mode:
            # Real & Imaginary
            if cur_mode=="cplx":
                if self.channel=="mono":
                    output = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                elif self.channel=="stereo":
                    output_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = torch.cat((output_l, output_r), axis=-1)
                if input.shape[2] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if self.n_fft % 2 == 0:
                    output = output[:, :-1]
                front_output_list.append(output.permute(0, 3, 1, 2))
            # Magnitude & Phase
            elif cur_mode=="mag":
                if self.channel=="mono":
                    cur_cplx = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = self.mag(cur_cplx).unsqueeze(-1)[..., 0:1]
                elif self.channel=="stereo":
                    cplx_l = torch.stft(input[:,0], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    cplx_r = torch.stft(input[:,1], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    mag_l = self.mag(cplx_l).unsqueeze(-1)
                    mag_r = self.mag(cplx_r).unsqueeze(-1)
                    output = torch.cat((mag_l, mag_r), axis=-1)

                if input.shape[-1] % round(self.n_fft/4) == 0:
                    output = output[:, :, :-1]
                if self.n_fft % 2 == 0: # discard highest frequency
                    output = output[:, 1:]
                front_output_list.append(output.permute(0, 3, 1, 2))

        # combine all demanded features
        if not front_output_list:
            raise NameError("NameError at FrontEnd: check using features for front-end")
        elif len(mode)!=1:
            for i, cur_output in enumerate(front_output_list):
                if i==0:
                    front_output = cur_output
                else:
                    front_output = torch.cat((front_output, cur_output), axis=1)
        else:
            front_output = front_output_list[0]
            
        return front_output


    def mag(self, cplx_input, eps=1e-07):
        mag_summed = cplx_input.pow(2.).sum(-1) + eps
        return mag_summed.pow(0.5)




class BackEnd(nn.Module):
    def __init__(self, channel='stereo', \
                        n_fft=2048, \
                        hop_length=None, \
                        win_length=None, \
                        window="hann", \
                        eps=1e-07, \
                        orig_freq=44100, \
                        new_freq=16000, \
                        device=torch.device("cpu")):
        super(BackEnd, self).__init__()
        self.device = device
        self.channel = channel
        self.n_fft = n_fft
        self.hop_length = n_fft//4 if hop_length==None else hop_length
        self.win_length = n_fft if win_length==None else win_length
        self.eps = eps
        if window=="hann":
            self.window = torch.hann_window(window_length=self.win_length, periodic=True).to(self.device)
        elif window=="hamming":
            self.window = torch.hamming_window(window_length=self.win_length, periodic=True).to(self.device)
        self.resample_func_8k = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=8000).to(self.device)
        self.resample_func = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq).to(self.device)

    def magphase_to_cplx(self, magphase_spec):
        real = magphase_spec[..., 0] * torch.cos(magphase_spec[..., 1])
        imaginary = magphase_spec[..., 0] * torch.sin(magphase_spec[..., 1])
        return torch.cat((real.unsqueeze(-1), imaginary.unsqueeze(-1)), dim=-1)


    def forward(self, input, phase, mode):
        # back-end function which convert output spectrograms into waveform
        # input shape : batch x channel x frequency x time
        # output shape : batch x channel x raw waveform

        # convert to shape : batch x frequency x time x channel
        input = input.permute(0, 2, 3, 1)
        # pad highest frequency
        pad = torch.zeros((input.shape[0], 1, input.shape[2], input.shape[3])).to(self.device)
        input = torch.cat((pad, input), dim=1)

        back_output_list = []
        channel_count = 0
        for i, cur_mode in enumerate(mode):
            # Real & Imaginary
            if cur_mode=="cplx":
                if self.channel=="mono":
                    output = ta_F.istft(input[...,channel_count:channel_count+2], n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window).unsqueeze(1)
                    channel_count += 2
                elif self.channel=="stereo":
                    cplx_spec = torch.cat([input[...,channel_count:channel_count+2], input[...,channel_count+2:channel_count+4]], dim=0)
                    output_wav = ta_F.istft(cplx_spec, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = torch.cat((output_wav[:output_wav.shape[0]//2].unsqueeze(1), output_wav[output_wav.shape[0]//2:].unsqueeze(1)), dim=1)
                    channel_count += 4
                back_output_list.append(output)
            # Magnitude & Phase
            elif cur_mode=="mag_phase" or cur_mode=="mag":
                if self.channel=="mono":
                    if cur_mode=="mag":
                        input_spec = torch.cat((input[...,channel_count:channel_count+1], phase), axis=-1)
                        channel_count += 1
                    else:
                        input_spec = input[...,channel_count:channel_count+2]
                        channel_count += 2
                    cplx_spec = self.magphase_to_cplx(input_spec)
                    output = ta_F.istft(cplx_spec, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window).unsqueeze(1)
                elif self.channel=="stereo":
                    if cur_mode=="mag":
                        input_spec_l = torch.cat((input[...,channel_count:channel_count+1], phase[...,0:1]), axis=-1)
                        input_spec_r = torch.cat((input[...,channel_count+1:channel_count+2], phase[...,1:2]), axis=-1)
                        channel_count += 2
                    else:
                        input_spec_l = input[...,channel_count:channel_count+2]
                        input_spec_r = input[...,channel_count+2:channel_count+4]
                        channel_count += 4
                    cplx_spec_l = self.magphase_to_cplx(input_spec_l)
                    cplx_spec_r = self.magphase_to_cplx(input_spec_r)
                    cplx_spec = torch.cat([cplx_spec_l, cplx_spec_r], dim=0)
                    output_wav = torch.istft(cplx_spec, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, window=self.window)
                    output = torch.cat((output_wav[:output_wav.shape[0]//2].unsqueeze(1), output_wav[output_wav.shape[0]//2:].unsqueeze(1)), dim=1)
                    channel_count += 4
                back_output_list.append(output)
            elif cur_mode=="griff":
                if self.channel=="mono":
                    output = self.griffin_lim(input.squeeze(-1), input.device).unsqueeze(1)
                    # output = self.griff(input.permute(0, 3, 1, 2))
                else:
                    output_l = self.griffin_lim(input[..., 0], input.device).unsqueeze(1)
                    output_r = self.griffin_lim(input[..., 1], input.device).unsqueeze(1)
                    output = torch.cat((output_l, output_r), axis=1)

            back_output_list.append(output)

        # combine all demanded feature outputs
        if not back_output_list:
            raise NameError("NameError at BackEnd: check using features for back-end")
        elif len(mode)!=1:
            for i, cur_output in enumerate(back_output_list):
                if i==0:
                    back_output = cur_output
                else:
                    back_output = torch.cat((back_output, cur_output), axis=1)
        else:
            back_output = back_output_list[0]
        
        return back_output


    def griffin_lim(self, l_est, gpu, n_iter=100):
        l_est = l_est.cpu().detach()

        l_est = torch.pow(l_est, 1/0.80)
        # l_est  [batch, channel, time]
        l_mag = l_est.unsqueeze(-1)
        l_phase = 2 * np.pi * torch.rand_like(l_mag) - np.pi
        real = l_mag * torch.cos(l_phase)
        imag = l_mag * torch.sin(l_phase)
        S = torch.cat((real, imag), axis=-1)
        S_mag = (real**2 + imag**2 + self.eps) ** 1/2
        for i in range(n_iter):
            x = ta_F.istft(S, n_fft=2048, hop_length=512, win_length=2048, window=torch.hann_window(2048))
            S_new = torch.stft(x, n_fft=2048, hop_length=512, win_length=2048, window=torch.hann_window(2048))
            S_new_phase = S_new/mag(S_new)
            S = S_mag * S_new_phase
        return x / torch.max(torch.abs(x))



if __name__ == '__main__':

    batch_size = 16
    channel = 2
    segment_length = 512*128*6
    input_wav = torch.rand((batch_size, channel, segment_length))

    mode = ["cplx", "mag"]
    fe = FrontEnd(channel="stereo")
    
    output = fe(input_wav, mode=mode)
    print(f"Input shape : {input_wav.shape}\nOutput shape : {output.shape}")
