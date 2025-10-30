
def loading_bar(step, t, config):
    if step % max(1, int(0.05 * config.num_steps)) == 0:
        progress = step / config.num_steps
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\rProgress: [{bar}] {progress*100:.1f}% | "
                f"Time: {t/3600:.1f}h / {config.t_end/3600:.1f}h",
                end='', flush=True)
        
def fancy_loading_bar(step, t, config, water_level=None, max_level=1.0):
    if step % max(1, int(0.05 * config.num_steps)) == 0:
        progress = step / config.num_steps
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)

        water_bar = ''
        if water_level is not None:
            height = int(10 * water_level / max_level)
            for i in range(10, 0, -1):
                if i <= height:
                    water_bar += "🌊\n"
                else:
                    water_bar += "  \n"

        print(f"\033[H\033[J", end='')  # clear screen
        print(f"Progress: [{bar}] {progress*100:.1f}% | "
              f"Time: {t/3600:.1f}h / {config.t_end/3600:.1f}h\n")
        if water_bar:
            print("Water level:")
            print(water_bar)

def colored_cell(value):
    # Map 0–1 to 21–231 (blue to white)
    color = int(21 + 210 * value)
    return f"\033[48;5;{color}m \033[0m"

if __name__ == "__main__":
    print(colored_cell(0.9) + " Low saturation")