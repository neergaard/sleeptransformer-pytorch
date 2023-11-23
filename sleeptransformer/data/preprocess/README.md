# <!-- omit from toc --> MrOS data processing pipeline

1. [Run montage coder](#run-montage-coder)
2. [Convert EDF to H5 files](#convert-edf-to-h5-files)
   1. [Arousals](#arousals)
   2. [Leg movements](#leg-movements)
   3. [Sleep disordered breathing](#sleep-disordered-breathing)
3. [H5 file structure](#h5-file-structure)

## Run montage coder
The following will read each EDF file and generate a list of all channel label names.
The user will then make a mapping to the desired channels by linking each available channel label name to a desired category.

```bash
python -m nsrr_data.utils.montage_coder -d data/raw \
                                        -o data/montage_code/montage_code.json \
                                        -c A1 A2 C3 C4 EOGL EOGR LChin RChin LegL LegR NasalP Thor Abdo
```

## Convert EDF to H5 files
This will convert each EDF file to an H5 file containing the signals and associated events

### Arousals
```bash
python -m nsrr_data.preprocessing -c mros \
                                  -d data/raw/mros \
                                  -o data/processed/mros/ar \
                                  --fs 128 \
                                  --duration 600 \  # Each data segment is 600 s long
                                  --overlap 300 \   # Data segments are overlapping by 300 s
                                  --event_type ar   # Only look at arousals
```

### Leg movements
```bash
python -m nsrr_data.preprocessing -c mros \
                                  -d data/raw/mros \
                                  -o data/processed/mros/lm \
                                  --fs 64 \
                                  --duration 600 \  # Each data segment is 600 s long
                                  --overlap 300 \   # Data segments are overlapping by 300 s
                                  --event_type lm   # Only look at leg movements
```

### Sleep disordered breathing
```bash
python -m nsrr_data.preprocessing -c mros \
                                  -d data/raw/mros \
                                  -o data/processed/mros/sdb \
                                  --fs 64 \
                                  --duration 600 \  # Each data segment is 600 s long
                                  --overlap 300 \   # Data segments are overlapping by 300 s
                                  --subjects 200 \
                                  --event_type sdb  # Only look at sleep disordered breathing
```

## H5 file structure
The H5 files can be loaded and viewed in Python using the `h5py` library:
```python
import h5py

f = h5py.File('path/to/file.h5')
print(f.keys())
```
Each file is structured like so:
```bash
f
├── f['data']
│   ├── f['data/channel_idx']
│   ├── f['data/fs']                                                # List of sampling frequencies after resampling
│   ├── f['data/fs_orig']                                           # List of original sampling frequencies
│   ├── f['data/scaled']
│   │   └── <HDF5 dataset "scaled": shape (N, C, T), type "<f8">    # Chunked dataset scaled by robust scaling each channel in C
│   └── f['data/unscaled']
│       └── <HDF5 dataset "unscaled": shape (N, C, T), type "<f8">  # Chunked unscaled dataset
├── f['events']
│   └── f['events/<event_type>']
│       ├── f['events/<event_type>/duration']
│       │   └── <HDF5 dataset "duration": shape (K,), type "<f8">   # for K number of events
│       └── f['events/<event_type>/start']
│           └── <HDF5 dataset "start": shape (K,), type "<f8">      # for K number of events
└── f['stages']
    └── <HDF5 dataset "stages": shape (S,), type "<i8">             # Sleep stage every second
```
