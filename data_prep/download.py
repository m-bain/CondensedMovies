import json
import os
from os.path import join as osj
import pandas as pd
import pdb

hosting_address = 'https://www.robots.ox.ac.uk/~vgg/research/condensed-movies/data'


def download_features(data_dir):
    cmd = 'wget {}/features.zip -P {}; unzip {}/features.zip -d {}'.format(hosting_address, data_dir, data_dir, data_dir)
    os.system(cmd)


def download_facetracks(data_dir):
    cmd = 'wget {}/facetracks.zip -P {}/facetracks.zip; unzip {}/facetracks.zip -d {}'.format(hosting_address, data_dir,
                                                                                              data_dir, data_dir)
    os.system(cmd)


def youtube_download(data_dir):
    id_dir = '../data/metadata/youtube-dl-dump'
    video_dir = osj(data_dir, 'videos')
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    for file in os.listdir(id_dir):
        upload_year = file.replace('.csv', '')
        video_dir_year = osj(video_dir, upload_year)
        if not os.path.exists(video_dir_year):
            os.makedirs(video_dir_year)

        output_fmt = osj(video_dir_year, '%(id)s.%(ext)s')
        id_fp = osj(id_dir, file)
        cmd = 'youtube-dl --config-location youtube-dl.conf -o "{}" -a "{}"'.format(output_fmt, id_fp)
        os.system(cmd)

    # trim advertisement outro from video.
    trim = None
    while trim not in ['y', 'n']:
        trim = str(input(
            "\nDo you want to trim the videos (y/n)?\nThis removes the advertisements (unrelated to the film), and only needs to be done once per download."))

        if trim not in ['y', 'n']:
            print('Please type "y" or "n"')

    if trim == "y":
        trim_video_outro(video_dir)

    # check for failed downloads: this can be due to...
    # i) geographical restrictions
    # ii) Too many requests error
    check_missing_vids(video_dir)


def trim_video_outro(video_dir, video_ext='.mkv'):
    duration_data = pd.read_csv('../data/metadata/durations.csv').set_index('videoid')

    tmp_fp = osj(video_dir, 'tmp' + video_ext)
    for root, subdir, files in os.walk(video_dir):
        for file in files:
            if file.endswith(video_ext) and file != 'tmp' + video_ext:
                videoid = file.split(video_ext)[0]
                pdb.set_trace()
                if videoid not in duration_data.index:
                    raise ValueError("Videoid not found, video files should be in format {VIDEOID}.mkv")

                video_fp = osj(root, file)
                new_duration = duration_data.loc[videoid]['duration']

                # create tmp for untrimmed
                os.system('cp {} {}'.format(video_fp, tmp_fp))
                cmd = 'ffmpeg -y -ss 0 -i {} -t {} -c copy {}'.format(tmp_fp, new_duration, video_fp)
                os.system(cmd)
                os.remove(tmp_fp)


def check_missing_vids(video_dir, video_ext='.mkv'):
    missing_ids = []
    clips_data = pd.read_csv('../data/metadata/clips.csv').set_index('videoid')
    for idx, row in clips_data.iterrows():
        videoid = row.name
        upload_year = row['upload_year']
        video_fp = osj(video_dir, str(int(upload_year)), videoid + video_ext)
        if not os.path.isfile(video_fp):
            missing_ids.append(videoid)

    success = (1 - len(missing_ids) / len(clips_data)) * 100
    print('=======================================================\n%.2f %% of clips downloaded successfully' % success)
    if success == 100:
        pass
    elif success < 100:
        print(
            '%d clips failed to download. This is likely due to geographical restrictions.\n-->Contact maxbain@robots.ox.ac.uk if this is an issue.' % len(
                missing_ids))

    with open('missing_videos.out', 'w') as fid:
        for mid in missing_ids:
            fid.write(mid + '\n')


def main():
    config = json.load(open('config.json', 'r'))
    data_dir = config['data_dir']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if config['features']:
        download_features(data_dir)
    if config['facetracks']:
        download_facetracks(data_dir)
    if config['src']:
        youtube_download(data_dir)


if __name__ == "__main__":
    main()
