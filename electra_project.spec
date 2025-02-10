
# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

block_cipher = None

# Obtenir le chemin absolu du répertoire racine du projet
BASE_DIR = os.path.abspath('.')

# Collecter tous les sous-modules nécessaires
hiddenimports = collect_submodules('network_reliability') + [
    'django.contrib.sessions',
    'django.contrib.contenttypes',
    'django.contrib.admin',
    'django.contrib.staticfiles',
    'django.contrib.messages',
    'django.template.context_processors',
    'django.contrib.auth.context_processors',
    'django.contrib.messages.context_processors',
    'django.template.loaders',
    'pyAgrum',
    'torch',  # Ajoutez torch si nécessaire
    'keras.src.backend.numpy',  # Ajoutez Keras backend numpy
    'jax'  # Ajoutez jax
]

# Collecter les fichiers de données nécessaires
datas = collect_data_files('django') + [
    (os.path.join(BASE_DIR, 'network_reliability/templates'), 'network_reliability/templates'),
    (os.path.join(BASE_DIR, 'static'), 'static'),
    (os.path.join(BASE_DIR, 'data'), 'data')
]

# Collecter les dépendances de NumPy
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all('numpy')

# Ajouter les dépendances de NumPy à l'analyse
datas += numpy_datas
binaries = numpy_binaries
hiddenimports += numpy_hiddenimports

a = Analysis(
    ['run_waitress.py'],  # Utilisez run_waitress.py comme script principal
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch.testing._internal.optests', 'tensorflow-plugins'],  # Exclure les modules non critiques
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='electra_project',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='electra_project',
)
