# User Handbook - Command Line

The SDL frontend (`veshiru-cmd`) and Qt frontend (`veshiru`) share the game-launch options below.
The native console options apply to the Qt frontend, not `veshiru --room`.

- `./veshiru <path>`: Running with a single argument and nothing else, will make the emulator look for the given file and load it, this behavior is similar to `veshiru-cmd`; allows dragging and dropping games into the application.
- `--debug/-d`: Enter debug mode, allow gdb stub at port `1234`
- `--config/-c`: Specify alternate configuration file.
- `--fullscreen/-f`: Use fullscreen.
- `--help/-h`: Display help.
- `--game/-g <path>`: Alternate way to specify what to load, overrides. However let it be noted that arguments that use `-` will be treated as options/ignored, if your game, for some reason, starts with `-`, in order to safely handle it you may need to specify it as an argument.
- `--multiplayer/-m`: Specify multiplayer options.
- `--program/-p`: Specify the program arguments to pass (optional).
- `--user/-u <number>`: Specify the user index.
- `--version/-v`: Display version and quit.
- `--input-profile/-i <name>`: Specifies input profile name to use (for player #0 only).
- `--null-render/-n`: Forces the usage of the "Null" render backend irrespective of settings.
- `--filter/-x`: Sets the debug log filter irrespective of settings.
- `--singlecore/-s`: Forces single-core regardless of settings.

Only the Qt frontend supports the following arguments:

- `-qlaunch`: Launch QLaunch.
- `-hlaunch`: Launch homebrew launcher `nx-hbloader`.
    - Requires a copy of Atmosphere to be extracted onto `sdmc`.
    - This is a shorthand for `<veshiru folder>/sdmc/atmosphere/hbl.nsp`.
- `-setup`: Launch setup applet.

## Qt frontend diagnostics (Windows)

`veshiru.exe` is a console-subsystem application with a Qt interface. From cmd.exe, PowerShell, or
Windows Terminal it uses the inherited standard handles. Explorer uses Windows' normal console
host, not a Veshiru terminal window. Use Ctrl+C or Ctrl+Break to request an orderly shutdown.

- `--console`: enable the standard-stream log (default). If unattached with no redirected output,
  this explicitly requests attachment to the parent console or a new system console.
- `--no-console`: mute the standard-stream log and detach the Windows console. The session log
  remains enabled. Windows can briefly open a console when launching a console executable from Explorer.
- `--log-level LEVEL`: minimum severity for all log sinks for this invocation, even if the saved
  configuration uses another level. Use `Trace`, `Debug`, `Info`, `Warning`, `Error`, or `Critical`.
- `--log-file PATH`: append UTF-8 session records to an additional disk file, even with `--no-console`.
  Relative paths are resolved at launch; the parent directory must exist. On Windows use a normal
  drive path, not a UNC address or device name. This extra file is not automatically rotated.

```bat
veshiru.exe > boot.log 2>&1
veshiru.exe --no-console --log-level Warning --log-file "C:\Logs\veshiru-boot.log"
```

Trace through Warning goes to stdout; Error and Critical go to stderr. Redirected output is plain
UTF-8 without terminal escapes. Under sustained overload or a blocked reader, the logger may omit
records rather than let a pipe block emulation. The session spool also has a size limit; the Debug
settings expose its location and the extended logging option. Keep logs private and review them
before sharing, even after redaction. **Save Diagnostic Log…** copies complete records already on
disk and applies redaction regardless of the saved username setting.

## Room host options

`veshiru` (provided with `--room`), and `veshiru-room` supports the following options as well:

- `-n/--room-name`: The name of the room.
- `-d/--room-description`: The room description.
- `-s/--bind-address`: The bind address for the room.
- `-p/--port`: The port used for the room.
- `-m/--max-members`: The maximum number of players for this room.
- `-w/--password`: The password for the room.
- `-g/--preferred-game`: The preferred game for this room.
- `-i/--preferred-game-id`: The preferred game-id for this room.
- `-u/--username`: The username used for announce.
- `-t/--token`: The token used for announce.
- `-a/--web-api-url`: Veshiru Web API url.
- `-b/--ban-list-file`: The file for storing the room ban list.
- `-l/--log-file`: The file for storing the room log.
- `-h/--help`: Display this help and exit.
- `-v/--version`: Output version information and exit.
