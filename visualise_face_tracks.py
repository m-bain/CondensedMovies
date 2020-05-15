# requires ffmpeg (2.8.15)
import os, cv2, pickle, argparse, random
from tqdm import tqdm
import pandas as pd
import pdb

track_colour_choices = [(75,25,230), (25,225,255), (75,180,60), (230,50,240), (240,240,70), (49,130,245), (180,30,145), (12,246,188), (216,99,67), (195,255,170), (255,190,230)]
random.shuffle(track_colour_choices)

def expandrect(ROI, extensionx, extensiony, shape):
    """expand the face detection bounding box"""

    width = ROI[2] - ROI[0]
    height = ROI[3] - ROI[1]
    #Length = (width + height) / 2
    centrepoint = [int(ROI[0]) + (width / 2), int(ROI[1]) + (height / 2)]
    x1 = int(centrepoint[0] - int((1 + extensionx) * width / 2))
    y1 = int(centrepoint[1] - int((1 + extensiony) * height / 2))
    x2 = int(centrepoint[0] + int((1 + extensionx) * width / 2))
    y2 = int(centrepoint[1] + int((1 + extensiony) * height / 2))

    x1 = max(1, x1)
    y1 = max(1, y1)
    x2 = min(x2, shape[1])
    y2 = min(y2, shape[0])

    return [x1, y1, x2, y2]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_ID', default='9YyC1Uq84v8', help='the ID of the video for face-track visualisation', type=str)
    parser.add_argument('--out_dir', default='figs', help='path to output directory for saving visualisation video', type=str)
    parser.add_argument('--data_dir', default='data/videos', help='path to video data directory', type=str)
    parser.add_argument('--face_track_dir', default='data/facetracks', help='path to data directory containing the face-tracks', type=str)

    args = parser.parse_args()

    # automatically get video year
    clips = pd.read_csv('metadata/clips.csv')
    target_clip = clips[clips['videoid'] == args.video_ID]
    if len(target_clip) == 0:
        raise Exception('video ID not found')
    video_year = str(int(target_clip['upload_year'].iloc[0]))

    # automatically get data directory
    
    # check that the output directory exists

    if not os.path.isdir(args.out_dir):
        raise Exception('path to output does not exist')

    # check that the path exists to the video

    if not os.path.isfile(os.path.join(args.data_dir, video_year, args.video_ID + '.mkv')):
        pdb.set_trace()
        raise Exception('path to video does not exist')

    # check that the path exists to the face-track

    if not os.path.isfile(os.path.join(args.face_track_dir, video_year,args.video_ID+'.mkvface_dets.pk')):
        raise Exception('path to face detections does not exist')

    # load the face-tracks

    with open(os.path.join(args.face_track_dir, video_year,args.video_ID+'.mkvface_dets.pk'), 'rb') as f:
        face_dets = pickle.load(f)
    with open(os.path.join(args.face_track_dir, video_year, args.video_ID + '.mkvdatabase.pk'), 'rb') as f:
        database = pickle.load(f)

    # extract the frames to the output directory

    if not os.path.isdir(os.path.join(args.out_dir, args.video_ID)):
        os.mkdir(os.path.join(args.out_dir, args.video_ID))
    else:
        os.system('rm -R '+ os.path.join(args.out_dir, args.video_ID))
        os.mkdir(os.path.join(args.out_dir, args.video_ID))

    Command = "ffmpeg -i " + os.path.join(args.data_dir, video_year, args.video_ID + '.mkv') + " -threads 1 -deinterlace -q:v 1 -s 640:360 -vf fps=25 " + os.path.join(args.out_dir,args.video_ID) + "/%05d.jpg"
    os.system(Command)
    extracted_frames = [f for f in os.listdir(os.path.join(args.out_dir, args.video_ID))]
    if len(extracted_frames) == 0:
        raise Exception('problem with frame extraction - check ffmpeg usage')

    if os.path.isfile(os.path.join(args.out_dir,'audio.mp3')):
        os.system('rm -R ' + os.path.join(args.out_dir, 'audio.mp3'))
    # extract the audio to the output directory
    audio_call = "ffmpeg -i " + os.path.join(args.data_dir, video_year, args.video_ID + '.mkv') +" "+ os.path.join(args.out_dir,'audio.mp3')
    os.system(audio_call)
    # for each track in the face-track, read and write the detection
    print('writing face tracks...')
    for track_ID, face_track_frames in enumerate(tqdm(database['index_into_facedetfile'])):

        for index in face_track_frames:

            frame = "%05d.jpg"%face_dets[index][0]

            image = cv2.imread(os.path.join(args.out_dir, args.video_ID, frame))

            ROI = [int(face_dets[index][1]), int(face_dets[index][2]), int(face_dets[index][1]+ face_dets[index][3]), int(face_dets[index][2]+face_dets[index][4])] # [x1, y1, x2, y2]

            track_colour = track_colour_choices[track_ID%len(track_colour_choices)]

            expand_rect = expandrect(ROI, 0.4, 0.4, image.shape) # expand the face detection for visualisation

            image = cv2.rectangle(image, (int(expand_rect[0]), int(expand_rect[1])), (int(expand_rect[2]), int(expand_rect[3])), track_colour,
                                  int(max(min(7, ((expand_rect[2] - expand_rect[0]) / 20)), 2))) # draw the bounding box
            image = cv2.putText(image, str(track_ID), (int(expand_rect[0]), int(expand_rect[3]) + 30), 0, 1, track_colour,
                                3)

            cv2.imwrite(os.path.join(args.out_dir, args.video_ID, frame), image)

    # make the output video
    FFMPEGCall = 'ffmpeg -r 25 -start_number 0 -i ' + os.path.join(args.out_dir, args.video_ID) + '/%05d.jpg -c:v libx264 -vf fps=25 -pix_fmt yuv420p ' + os.path.join(args.out_dir, args.video_ID+ '.mp4')
    os.system(FFMPEGCall)

    # delete the frames

    os.system('rm -R '+ os.path.join(args.out_dir, args.video_ID))

    # add audio

    audio_call = "ffmpeg -i "+os.path.join(args.out_dir, args.video_ID+ '.mp4') + " -i "+os.path.join(args.out_dir,'audio.mp3')+" -c:v libx264 -c:a libvorbis -shortest " + os.path.join(args.out_dir, args.video_ID+ '.mkv')
    os.system(audio_call)
    os.system('rm '+os.path.join(args.out_dir, args.video_ID+ '.mp4'))
    os.system('rm -R ' + os.path.join(args.out_dir,'audio.mp3'))
