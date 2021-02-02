import logomaker


def plot_saliency(saliency, region=range(0, 1000)):
    pwm_df = pd.DataFrame(
        data=saliency[:,region].numpy().transpose(),
        columns=("A","C","G","T")
    )
    ax = logomaker.Logo(pwm_df)
    return ax