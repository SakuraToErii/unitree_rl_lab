# Keyboard Joystick

`unitree_mujoco` can simulate a Unitree wireless controller with the local keyboard. This is useful when there is no USB or Bluetooth gamepad.

## Enable It

Edit `/home/ordis/projects/unitree_mujoco/simulate/config.yaml`:

```yaml
use_joystick: 1
joystick_type: "keyboard"
joystick_device: "keyboard"
```

`joystick_device: "keyboard"` scans `/dev/input/event*`. If it cannot find or read the keyboard device, set a concrete device path instead:

```yaml
joystick_device: "/dev/input/event3"
```

## Key Mapping

| Unitree input | Keyboard |
| --- | --- |
| Left stick | `W` `A` `S` `D` |
| Right stick | `I` `J` `K` `L` |
| D-pad Up / Left / Down / Right | `T` / `F` / `G` / `H` |
| `A` button | `Z` |
| `B` button | `X` |
| `X` button | `C` |
| `Y` button | `V` |
| `L1` / `R1` | `Q` / `E` |
| `L2` / `R2` | `1` / `2` |
| `Select` / `Start` | `Tab` / `Enter` |
| `F1` / `F2` | `F1` / `F2` |
| `LS` / `RS` | Left Shift / Right Shift |

## Permissions

The simulator reads Linux input events directly. If it prints a keyboard open error, check that your user can read `/dev/input/event*`.

On most Linux desktops:

```bash
groups
sudo usermod -aG input "$USER"
```

Then log out and back in.

## Notes

The code uses Xbox-style names internally: `LB` / `RB` map to `L1` / `R1`, and `LT` / `RT` map to `L2` / `R2`.

Physical arrow keys are reserved by `unitree_mujoco` for the elastic band. In `g1_ctrl`, `L2 + Up` means Unitree D-pad Up, so press `1 + T`.

`wireless_controller` publishes `lx`, `ly`, `rx`, `ry`, and the Unitree 16-bit key field. `LS` and `RS` are mapped in local joystick state, but the current Unitree key packet does not serialize them into `wireless_controller.keys`.
