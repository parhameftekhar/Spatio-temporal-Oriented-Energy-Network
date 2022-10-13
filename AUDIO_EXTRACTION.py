import moviepy.editor as mp

file_name = "keyboard_103"
clip = mp.VideoFileClip(r"{}.mp4".format(file_name))
clip.audio.write_audiofile(r"{}.mp3".format(file_name))




