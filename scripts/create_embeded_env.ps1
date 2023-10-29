$PY_ZIP_SAVE_AS="./python_embeded.zip"
$PY_EMBED_DIR="./python_embeded"
$GIT_REPO="https://github.com/frimin/CLIPImageSearchWebUI"
$REPO_DIR="webui"

if (!(Test-Path -Path "$REPO_DIR/.git")) {
    git clone $GIT_REPO "$REPO_DIR"
}

if (!(Test-Path -Path $PY_EMBED_DIR)) {
    if (!(Test-Path -Path $PY_ZIP_SAVE_AS)) {
        Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.9/python-3.10.9-embed-amd64.zip" -OutFile $PY_ZIP_SAVE_AS
    }

    Expand-Archive $PY_ZIP_SAVE_AS -DestinationPath $PY_EMBED_DIR
}

if (!(Test-Path -Path "$PY_EMBED_DIR/get-pip.py")) {
    Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile "$PY_EMBED_DIR/get-pip.py"
}

& $PY_EMBED_DIR/python.exe "$PY_EMBED_DIR/get-pip.py"

if (Test-Path -Path "$PY_EMBED_DIR/python310._pth") {
    Move-Item -Path "$PY_EMBED_DIR/python310._pth" "$PY_EMBED_DIR/python310.pth"
}

if (!(Test-Path $PY_EMBED_DIR/DLLs)) {
    New-Item -ItemType Directory $PY_EMBED_DIR/DLLs
}

& $PY_EMBED_DIR/python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
& $PY_EMBED_DIR/python.exe -m pip install -r $REPO_DIR/requirements.txt
& $PY_EMBED_DIR/python.exe -m pip install -e $REPO_DIR

if (!(Test-Path run_embeded.bat)) {
    Copy-Item CLIPImageSearchWebUI/scripts/run_embeded.bat run.bat
}

Remove-Item $PY_ZIP_SAVE_AS