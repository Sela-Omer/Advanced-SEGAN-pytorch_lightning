from matplotlib import pyplot as plt


def plot_waveforms(noisy_audio_waveform, clean_audio_waveform):
    """
    Plots the waveforms of the noisy and clean audio.

    Args:
        noisy_audio_waveform (torch.Tensor): The waveform of the noisy audio.
        clean_audio_waveform (torch.Tensor): The waveform of the clean audio.

    Returns:
        None
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 3))

    # Plot clean audio
    axes[0].plot(clean_audio_waveform.t().numpy())
    axes[0].set_title("Clean Audio Waveform")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Amplitude")

    # Plot noisy audio
    axes[1].plot(noisy_audio_waveform.t().numpy())
    axes[1].set_title("Noisy Audio Waveform")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Amplitude")

    # Plot both audios for comparison
    axes[2].plot(clean_audio_waveform.t().numpy(), label="Clean Audio")
    axes[2].plot(noisy_audio_waveform.t().numpy(), label="Noisy Audio", alpha=0.5)
    axes[2].set_title("Combined Audio Waveforms")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
