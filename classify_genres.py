import os
from audio_classify import train_and_test


music_path = '/home/deniz/Music'

# Ambient music
ambt_albums = [
    'Brian Eno/1 Tracks and Traces (1976)', 
    'Brian Eno/2 Cluster & Eno (1977)', 
    'Brian Eno/3 After the Heat (1978)', 
    'Brian Eno/4 Begegnungen 1 (1984)', 
    'Brian Eno/5 Begegnungen 2 (1985)',
    'Ashra/New Age Of Earth 1977',
    'Cluster/Cluster 1971',
    'Harmonia/Musik von Harmonia 1973'
]
ambt_paths = [os.path.join(music_path,  path) for path in ambt_albums]

# Jazz music
jazz_albums = [
    # 'John Coltrane/Blue Train 1958',
    'John Coltrane/A Love Supreme 1965',
    'John Coltrane/Giant Steps 1960', 
    # 'John Coltrane/My Favorite Things 1961',
    'Charles Mingus Hits',
    'Duke Ellington',
    'Louis Armstrong/Disc 1',
    # 'Louis Armstrong/Disc 2',
]
jazz_paths = [os.path.join(music_path, path) for path in jazz_albums]

# Rock music
rock_albums = [
    'The Beatles/Revolver 1966',
    # 'The Beatles/Sgt Peppers Lonely Hearts Club Band 1967',
    'The Beatles/The White Album 1968/Disc 1/',
    # 'The Beatles/The White Album 1968/Disc 2/',  
    'The Beatles/Abbey Road 1969',
    'The Animals/Disc 1',
    # 'The Animals/Disc 2',
    'The Kinks/cd1/',
    # 'The Kinks/cd2',
    'The Who/The Who'
]
rock_paths = [os.path.join(music_path, path) for path in rock_albums]

title = 'genre/full1_'
model = train_and_test(title, ambt_paths, jazz_paths, rock_paths)
