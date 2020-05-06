import pandas as pd
import pdb
import os

years = range(2011, 2020)
exp_dir = '/scratch/shared/slow/maxbain/datasets/MovieAlignment/clips'
metadata_dir = os.path.join(exp_dir, 'metadata')
exp_csv = os.path.join(metadata_dir, 'missing')
data_dir = os.path.join(exp_dir, 'data')

def get_list_of_vid_ids(directory, ext='.mkv'):
    vid_ids = []
    for file in os.listdir(directory):
        if file.endswith(ext):
            # Vid file format is %(title)s-(%(duration)ss)[%(resolution)s][%(id)s].%(ext)s"
            # For %(id)s we want the last substring enclosed by []
            vid_id = file.split('[')[-1]
            vid_id = vid_id.split(']')[0]
            vid_ids.append(vid_id)

    return set(vid_ids)

for year in years:
    video_id = pd.read_csv(os.path.join(metadata_dir, str(year) + '.csv'), sep=';')
    video_id = set([x[0] for x in video_id.values])
    downloaded_ids = get_list_of_vid_ids(os.path.join(data_dir, str(year)))
    missing_ids = video_id.difference(downloaded_ids)
    pd.DataFrame(list(missing_ids)).to_csv(os.path.join(exp_csv, str(year) + '.csv'), index=False, header=False)
