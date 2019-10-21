import os
from audio_classify import train_and_test


music_path = '/home/deniz/Music'

# Ambient music by Brian Eno
ambt_albums = [
    'Brian Eno/1 Tracks and Traces (1976)',
    'Brian Eno/2 Cluster & Eno (1977)',
    'Brian Eno/3 After the Heat (1978)',
    'Brian Eno/4 Begegnungen 1 (1984)',
    'Brian Eno/5 Begegnungen 2 (1985)'
]
ambt_paths = [os.path.join(music_path,  path) for path in ambt_albums]

# Jazz music by John Coltrane
jazz_albums = [
    'John Coltrane/Blue Train 1958',
    'John Coltrane/A Love Supreme 1965',
    'John Coltrane/Giant Steps 1960',
    'John Coltrane/My Favorite Things 1961'
]
jazz_paths = [os.path.join(music_path, path) for path in jazz_albums]

# Rock music by The Beatles
rock_albums = [
    'The Beatles/Revolver 1966',
    'The Beatles/Sgt Peppers Lonely Hearts Club Band 1967',
    'The Beatles/The White Album 1968/Disc 1/',
    'The Beatles/The White Album 1968/Disc 2/',
    'The Beatles/Abbey Road 1969'
]
rock_paths = [os.path.join(music_path, path) for path in rock_albums]

title = 'bands/full5_'
models, acc = train_and_test(title, ambt_paths, jazz_paths, rock_paths)

