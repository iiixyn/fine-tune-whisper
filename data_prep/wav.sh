for IDX in {0..0}
do
    ffmpeg -i vid$IDX.mp3 -acodec pcm_s16le -ac 1 -ar 16000 vid$IDX.wav
done