
def loading_bar(step, t, config):
    if step % max(1, int(0.05 * config.num_steps)) == 0:
        progress = step / config.num_steps
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                f"Time: {t/3600:.1f}h / {config.t_end/3600:.1f}h",
                end='', flush=True)