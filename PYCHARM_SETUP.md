# PyCharm Docker Setup Anleitung

## Docker Server in PyCharm konfigurieren:

1. **File → Settings → Build, Execution, Deployment → Docker**
2. **Klicken Sie auf "+" um einen neuen Docker-Server hinzuzufügen**
3. **Wählen Sie "Docker for Windows" oder "Docker Desktop"**
4. **Server URL:** `npipe://./pipe/docker_engine` (für Windows)
5. **Klicken Sie "OK"**

## Alternative Server URLs falls die erste nicht funktioniert:
- `tcp://localhost:2375` (falls TCP aktiviert)
- `unix:///var/run/docker.sock` (für WSL/Linux)

## Run Configuration korrigieren:
1. **Run → Edit Configurations**
2. **Bei "Docker Compose Up" → Server:** Den neu erstellten Docker-Server auswählen
3. **Apply → OK**

## Testen:
- Wählen Sie "Docker Compose Up" aus dem Dropdown
- Klicken Sie auf Run
