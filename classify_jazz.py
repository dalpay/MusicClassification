import os
from audio_classify import train_and_test


music_path = '/home/deniz/Music'

# Charles Mingus
mings_albums = [
    'Charles Mingus/1960 Mingus - Mingus!  72.1mb',
    'Charles Mingus/1960 Mingus - Mysterious Blues  65.7mb',
    'Charles Mingus/1960 Mingus Presents Mingus  84.7mb',
    'Charles Mingus/1961 Mingus - Reincarnation of a Lovebird  38.5mb',
    'Charles Mingus/1974 Mingus - Changes One  49.9mb'
]
mings_paths = [os.path.join(music_path, path) for path in mings_albums]

# John Coltrane
coltr_albums = [
    'John Coltrane/Blue Train 1958',
    'John Coltrane/A Love Supreme 1965',
    'John Coltrane/Giant Steps 1960', 
    'John Coltrane/My Favorite Things 1961'
]
coltr_paths = [os.path.join(music_path, path) for path in coltr_albums]

# Miles Davis
davis_albums = [
    'Miles Davis/Kind of Blue 1959 CD 1',
    'Miles Davis/Kind of Blue 1959 CD 2',
    'Miles Davis/Sketches of Spain 1960'
]
davis_paths = [os.path.join(music_path, path) for path in davis_albums]

title = 'jazz/full2_'
model = train_and_test(title, mings_paths, coltr_paths, davis_paths)
