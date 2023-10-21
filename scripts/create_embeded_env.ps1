$PY_ZIP_SAVE_AS="./python_embeded.zip"
$PY_EMBED_DIR="./python_embeded"

if (!(Test-Path $PY_EMBED_DIR)) {
    if (!(Test-Path $PY_ZIP_SAVE_AS)) {
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip" -OutFile $PY_ZIP_SAVE_AS
    }

    Expand-Archive $PY_ZIP_SAVE_AS -DestinationPath $PY_EMBED_DIR
}

if (!(Test-Path "$PY_EMBED_DIR/get-pip.py")) {
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$PY_EMBED_DIR/get-pip.py"
}

& $PY_EMBED_DIR/python.exe "$PY_EMBED_DIR/get-pip.py"

if (Test-Path "$PY_EMBED_DIR/python310._pth") {
    Move-Item "$PY_EMBED_DIR/python310._pth" "$PY_EMBED_DIR/python310.pth"
}

if (!(Test-Path $PY_EMBED_DIR/DLLs)) {
    New-Item -ItemType Directory $PY_EMBED_DIR/DLLs
}

& $PY_EMBED_DIR/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
& $PY_EMBED_DIR/python.exe -m pip install -r requirements.txt
& $PY_EMBED_DIR/python.exe -m pip install -e .

if (!(Test-Path run_embeded.bat)) {
    Copy-Item scripts/run_embeded.bat run_embeded.bat
}