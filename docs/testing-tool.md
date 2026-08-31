# testing-tool

## How to Run

`testing-tool.py` accepts the following command-line arguments:

- `-h`, `--help`: Print help message.
- `-c CONFIG`, `--config CONFIG`: Use `CONFIG` as the **config file**.
- `-i INPUT`, `--input INPUT`: Use `INPUT` as the **input file** (map file).
- `-l LOG`, `--log LOG`: Use `LOG` as the **log file**.
- `-s`, `--stdio`: If no **input file** or **log file** is specified, use standard I/O instead.
- `-a EXEC1, --exec1 EXEC1`: Use `EXEC1` as the execution command for the LEFT player.
- `-b EXEC2, --exec2 EXEC2`: Use `EXEC2` as the execution command for the RIGHT player.
- `--seed S`: If omitted, a random seed is used.
- `--NP NP`: Use `NP` as the number of zones in the right half. The total number of zones will be $2NP+1$.
- `--KP KP`: Use `KP` as the number of neutral strongholds in the right half. The total number of strongholds will be $2KP+1$.

**Note: `--NP` and `--KP` are generation parameters equal to half the actual `N` and `K` values in the map file's first line.**
**In other words, `NP` and `KP` are the number of zones and neutral strongholds on one half of the battlefield.**
The generated actual `N` must satisfy 51 ≤ N ≤ 109, and the actual `K` must satisfy $\lceil 0.15N \rceil$ ≤ K ≤ $\lfloor 0.2N \rfloor$.

A map can be provided in one of three ways:
1. Use a pre-generated map file via the `INPUT` key in `config.ini` or the `-i INPUT` option
2. Generate a random map via the `SEED` key in `config.ini` or the `--seed S` option
3. Generate a random map of a specified size via `NP` and `KP` in `config.ini` or the `--NP NP` and `--KP KP` options

If multiple options are specified, the earlier one takes higher priority. If none are specified, a random seed is used to generate the map.

For example, to use `input.txt` as the input file, `log.txt` as the log file, `python3 sample-code.py P1` as the LEFT player's command, and `python3 sample-code.py P2` as the RIGHT player's command, run:

```bash
python3 testing-tool.py -i input.txt -l log.txt -a "python3 sample-code.py P1" -b "python3 sample-code.py P2"
```

Or you can generate a map on the fly using `--seed`:

```bash
python3 testing-tool.py --seed 42 -l log.txt -a "python3 sample-code.py P1" -b "python3 sample-code.py P2"
```

### Config File

The config file is a convenient alternative to command-line arguments. It supports the following keys:

```
INPUT=<path to input file>
LOG=<path to log file>
EXEC1=<execution command for the LEFT player>
EXEC2=<execution command for the RIGHT player>
SEED=<map generation seed>
NP=<number of zones in the right half>
KP=<number of neutral strongholds in the right half>
```

If a command-line argument conflicts with a config file value, the command-line argument takes priority.

For example, the run command above can be written as a config file as follows:

```
INPUT=input.txt
LOG=log.txt
EXEC1=python3 sample-code.py P1
EXEC2=python3 sample-code.py P2
```

To use seed-based map generation instead of a pre-generated map file, remove the `INPUT` line and specify `SEED` or the (`NP`, `KP`) pair.
If `SEED` is present, the map size is determined automatically from the seed. If both `NP` and `KP` are also specified, a map of that size is generated.
If `SEED` is absent, a random seed is used.

```
SEED=42
LOG=log.txt
EXEC1=python3 sample-code.py P1
EXEC2=python3 sample-code.py P2
```

Then run with:

```bash
python3 testing-tool.py -c config.ini
```

### Input File (Map File)

The input file describes the **battlefield layout** for the game. It has the following format:

```
N K
x_0 x_1 ... x_{N-1}
y_0 y_1 ... y_{N-1}
p_0 p_1 ... p_{K-1}
a_0 b_{0,0} b_{0,1} ... b_{0,a_0-1}
a_1 b_{1,0} b_{1,1} ... b_{1,a_1-1}
...
a_{N-1} b_{N-1,0} ... b_{N-1,a_{N-1}-1}
```

- First line: number of zones `N` and number of strongholds `K`.
- Second line: `x` coordinates of each zone's center (`N` integers).
- Third line: `y` coordinates of each zone's center (`N` integers).
- Fourth line: zone indices of the `K` neutral strongholds (in ascending order).
- Following `N` lines: for each zone `i`, the number of adjacent zones `a_i` followed by the adjacent zone indices `b_{i,*}`.

### Log File


```
[LEFT "COMMAND: <exec1>"]
[RIGHT "COMMAND: <exec2>"]
MAP
N K
x_0 x_1 ... x_{N-1}
y_0 y_1 ... y_{N-1}
STRONGHOLDS p_0 p_1 ... p_{K-1}
a_0 b_{0,0} b_{0,1} ...
...
a_{N-1} b_{N-1,0} ...
END MAP
TURN t
COMMAND LEFT START
<commands submitted by LEFT>
COMMAND LEFT END
COMMAND RIGHT START
<commands submitted by RIGHT>
COMMAND RIGHT END
TURN t RESULT
TIME LEFT <ms> <token> RIGHT <ms> <token>
<UPGRADE/TRAIN/MOVE/DAMAGE/SIEGE result lines>
END TURN t
...
RESULT <LEFT_WIN/RIGHT_WIN/DRAW> <HQ_DESTROYED/TURN_LIMIT/WA>
```

- `[LEFT ...]`, `[RIGHT ...]`: Shows the commands used to launch LEFT and RIGHT.
- `MAP` through `END MAP`: Records the battlefield layout, same as the map file. The stronghold line starts with `STRONGHOLDS ...`.
- `TURN t` through `END TURN t`: Records the commands submitted and their results for day `t`.
- `COMMAND <LEFT/RIGHT> START` through `COMMAND <LEFT/RIGHT> END`: Command lines submitted by the given player. If no commands were submitted, these two lines appear consecutively with nothing in between.
- After `TURN t RESULT`: Records time usage, remaining tokens, and UPGRADE/TRAIN/MOVE/DAMAGE/SIEGE results. Result types that did not occur are omitted.
- `DAMAGE <CAUSE> <warrior_id> <damage>`: A warrior took damage. `CAUSE` is `TURRET`, `COMBAT`, or `HUNGER`. `damage` is the amount of HP lost (not remaining HP).
- `SIEGE <side> <region> <damage>`: A building took siege damage. `damage` is the amount of HP lost (not remaining HP).
- `RESULT <LEFT_WIN/RIGHT_WIN/DRAW> <HQ_DESTROYED/TURN_LIMIT/WA>`: Game result. The reason for ending is one of: HQ destroyed (`HQ_DESTROYED`), day 200 reached (`TURN_LIMIT`), or wrong answer / timeout (`WA`).
- `# Debug <LEFT/RIGHT>: <msg>`: A line printed by LEFT or RIGHT to standard error (stderr).
