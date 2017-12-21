package com.mindorks.tensorflowexample;

import android.Manifest;
import android.content.Context;
import android.content.DialogInterface;
import android.content.pm.PackageManager;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaPlayer;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.support.annotation.NonNull;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.util.Pair;
import android.view.View;
import android.widget.Button;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.locks.ReentrantLock;

public class MainActivity extends AppCompatActivity {

    private static final String DBG_TAG = "MAIN";
    public static final int VOLUME_CHANGE_TIMER = 30000;

    private Context mContext = this;

    // Tensor Flow constant
    private static final int SAMPLE_RATE = 16000;
    private static final int SAMPLE_DURATION_MS = 1000;
    private static final int RECORDING_LENGTH = (int) (SAMPLE_RATE * SAMPLE_DURATION_MS / 1000);
    private static final long AVERAGE_WINDOW_DURATION_MS = 500;
    private static final float DETECTION_THRESHOLD = 0.70f;
    private static final int SUPPRESSION_MS = 1500;
    private static final int MINIMUM_COUNT = 3;
    private static final long MINIMUM_TIME_BETWEEN_SAMPLES_MS = 30;
    private static final String LABEL_FILENAME = "file:///android_asset/conv_actions_labels.txt";
    private static final String MODEL_FILENAME = "file:///android_asset/conv_actions_frozen.pb";
    private static final String INPUT_DATA_NAME = "decoded_sample_data:0";
    private static final String SAMPLE_RATE_NAME = "decoded_sample_data:1";
    private static final String OUTPUT_SCORES_NAME = "labels_softmax";

    // My additional constant
    private static final int MY_PERMISSIONS_REQUEST_RECORD_AUDIO = 100;
    private static final long DB_WINDOW_MS = 5000;
    private static final int NUM_SCENARIOS = 4;
    public static final int DB_THRESHOLD_SILENCE = 45;
    public static final int DB_THRESHOLD_SINGLE_VOICE = 75;
    private static final int FREQ_MAX_COUNTER = 6;
    public static final int FFT_INPUT_LENGTH = 1024;

    // Layouts
    private Button playButton;
    private MediaPlayer mediaPlayer;
    private SeekBar controller;
    private TextView volumeTextView;
    private TextView[] tvScenarios;
    private TextView timerDisplay;

    // My additional working variables
    private static final Double SINGLE_VOICE_DB_RANGE = 5.0;
    private static final Double SINGLE_VOICE_DB_RATIO = 0.7;
    private Deque<Pair<Long, Double>> previousDbResults = new ArrayDeque<>();
    private Double averageDb = 0.0;
    private Double majorityDbRatio = 0.0;
    private float currentVolume;
    private boolean isPlaying = false;
    private boolean isRunningTensorFlow = false;
    private int[] countScenarios;
    private int[] countScenariosLastOneMinute;
    private int timerCount = 0;
    private Handler handlerTimer;
    private Runnable runnableTimer;


    // Working variables.
    short[] recordingBuffer = new short[RECORDING_LENGTH];
    int recordingOffset = 0;
    boolean shouldContinue = true;
    private Thread recordingThread;
    boolean shouldContinueRecognition = true;
    private Thread recognitionThread;
    private final ReentrantLock recordingBufferLock = new ReentrantLock();
    private TensorFlowInferenceInterface inferenceInterface;
    private List<String> labels = new ArrayList<String>();
    private RecognizeCommands recognizeCommands = null;
    private int fftCounter = 0;
    private int freqCounter = 0;
    private int freqInRangeCounter = 0;
    private int freqInRangeFlag = 2;
    private int previousScenario = -1, currentScenario = -1;
    private Handler setupVolumeHandler;
    private Runnable setupVolumeRunnable;



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // Setup layout
        setupLayout();

        // Request microphone permission
        if (!checkPermission()) requestMicrophonePermission();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        if (mediaPlayer != null) mediaPlayer.release();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        switch (requestCode) {

            case MY_PERMISSIONS_REQUEST_RECORD_AUDIO: {
                Log.d(DBG_TAG, Arrays.asList(permissions).toString());
                Log.d(DBG_TAG, Arrays.asList(grantResults).toString());
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    Log.d(DBG_TAG, "Granted to record");
                } else {
                    Log.d(DBG_TAG, "No permission on audio recording");
                }
            }
            break;
        }
    }

    private boolean checkPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                        == PackageManager.PERMISSION_GRANTED &&
                ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                        == PackageManager.PERMISSION_GRANTED;
    }


    private void requestMicrophonePermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[]{android.Manifest.permission.RECORD_AUDIO},
                    MY_PERMISSIONS_REQUEST_RECORD_AUDIO);
        }
    }


    private void setupLayout() {

        // Demo button
        findViewById(R.id.tenser_flow_demo).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isRunningTensorFlow) {
                    Log.d(DBG_TAG, "Stop Tensor Flow");
                    stopTensorFlow();
                    isRunningTensorFlow = false;
                    ((TextView) v).setText("Start Detection");
                } else {
                    Log.d(DBG_TAG, "Start running Tensor Flow");
                    startTensorFlow();
                    isRunningTensorFlow = true;
                    ((TextView) v).setText("Stop");
                }
            }
        });

        // Play music button
        playButton = (Button) findViewById(R.id.btn_play_music);
        playButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (isPlaying) {
                    mediaPlayer.pause();
                    isPlaying = false;
                    playButton.setText(R.string.player_play);
                } else {
                    mediaPlayer.start();
                    isPlaying = true;
                    playButton.setText(R.string.player_pause);
                }
            }
        });

        // Media player
        mediaPlayer = MediaPlayer.create(this, R.raw.song);
        mediaPlayer.setVolume(currentVolume, currentVolume);
        mediaPlayer.setOnCompletionListener(new MediaPlayer.OnCompletionListener() {
            @Override
            public void onCompletion(MediaPlayer mp) {
                mediaPlayer.start();
            }

        });

        // Volume controller
        controller = (SeekBar) findViewById(R.id.controller_volume);
        volumeTextView = (TextView) findViewById(R.id.text_volume);
        volumeTextView.setText(controller.getProgress() + " / 100");
        currentVolume = (float) controller.getProgress() / 100;
        mediaPlayer.setVolume(currentVolume, currentVolume);
        controller.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                currentVolume = (float) controller.getProgress() / 100;
                volumeTextView.setText(controller.getProgress() + " / 100");
                mediaPlayer.setVolume(currentVolume, currentVolume);
                previousScenario = currentScenario = 0;
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        // Update volume every 1 minute
        setupVolumeHandler = new Handler();
        setupVolumeRunnable = new Runnable() {
            @Override
            public void run() {
                currentVolume = (float) controller.getProgress() / 100;
                currentScenario = 0;
                for(int i = 1; i < NUM_SCENARIOS; i++) {
                    if(countScenariosLastOneMinute[i] > countScenariosLastOneMinute[currentScenario]){
                        currentScenario = i;
                    }
                }
                Log.d(DBG_TAG, "previousScenario = " + previousScenario + ", currentScenario = " + currentScenario);
                // Scenario silence/single --> crowd/Ambience ==> Volume up
                if(currentVolume < 1 && (previousScenario == 0 || previousScenario == 1) &&
                        (currentScenario == 2 || currentScenario == 3)){
//                    currentVolume = Math.min(currentVolume + 0.2f, 1);
                    AlertDialog changeVolumeDialog = new AlertDialog.Builder(mContext)
                            .setCancelable(false)
                            .setTitle("Volume up")
                            .setMessage("We detected you entered a noise environment. Do you want to volume up?")
                            .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    currentVolume = Math.max(currentVolume + 0.2f, 0);
                                    volumeTextView.setText(String.format("%s / 100", Integer.toString((int) (currentVolume * 100))));
                                    mediaPlayer.setVolume(currentVolume,currentVolume);
                                    controller.setProgress((int) (currentVolume * 100));
                                    for(int i = 0; i < countScenariosLastOneMinute.length; i++){
                                        countScenariosLastOneMinute[i] = 0;
                                    }
                                    setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
                                    previousScenario = currentScenario = 0;
                                }
                            })
                            .setNeutralButton("Skip", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
                                    previousScenario = currentScenario = 0;
                                }
                            })
                            .setNegativeButton("Never", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                }
                            })
                            .create();
                    changeVolumeDialog.show();
                    Toast.makeText(mContext, "Volume up!!", Toast.LENGTH_SHORT).show();
                }
                else if(currentVolume > 0 && (previousScenario == 2 || previousScenario == 3) &&
                        (currentScenario == 0 || currentScenario == 1)){
//                    currentVolume = Math.max(currentVolume - 0.2f, 0);
                    AlertDialog changeVolumeDialog = new AlertDialog.Builder(mContext)
                            .setCancelable(false)
                            .setTitle("Volume Down")
                            .setMessage("We detected you entered a quite environment. Do you want to volume down?")
                            .setPositiveButton("Ok", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    currentVolume = Math.max(currentVolume - 0.2f, 0);
                                    volumeTextView.setText(String.format("%s / 100", Integer.toString((int) (currentVolume * 100))));
                                    mediaPlayer.setVolume(currentVolume,currentVolume);
                                    controller.setProgress((int) (currentVolume * 100));
                                    for(int i = 0; i < countScenariosLastOneMinute.length; i++){
                                        countScenariosLastOneMinute[i] = 0;
                                    }
                                    setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
                                    previousScenario = currentScenario = 0;
                                }
                            })
                            .setNeutralButton("Skip", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                    setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
                                    previousScenario = currentScenario = 0;
                                }
                            })
                            .setNegativeButton("Never", new DialogInterface.OnClickListener() {
                                @Override
                                public void onClick(DialogInterface dialog, int which) {
                                }
                            })
                            .create();
                    changeVolumeDialog.show();
                    Toast.makeText(mContext, "Volume down!!", Toast.LENGTH_SHORT).show();
                }
                else{
                    setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
                }
                previousScenario = currentScenario;

                Log.d(DBG_TAG, "currentVolume = " + currentVolume);
//                volumeTextView.setText(String.format("%s / 100", Integer.toString((int) (currentVolume * 100))));
//                mediaPlayer.setVolume(currentVolume,currentVolume);
//                controller.setProgress((int) (currentVolume * 100));
//                for(int i = 0; i < countScenariosLastOneMinute.length; i++){
//                    countScenariosLastOneMinute[i] = 0;
//                }
//                setupVolumeHandler.postDelayed(setupVolumeRunnable, VOLUME_CHANGE_TIMER);
            }
        };


        // Setup textview
        countScenarios = new int[NUM_SCENARIOS];
        countScenariosLastOneMinute = new int[NUM_SCENARIOS];
        tvScenarios = new TextView[NUM_SCENARIOS];
        tvScenarios[0] = (TextView) findViewById(R.id.silence_count);
        tvScenarios[1] = (TextView) findViewById(R.id.single_voice_count);
        tvScenarios[2] = (TextView) findViewById(R.id.crowd_voice_count);
        tvScenarios[3] = (TextView) findViewById(R.id.ambience_noise_count);

        // Timer
        timerDisplay = (TextView) findViewById(R.id.timer_tensor_flow);
        timerDisplay.setText("00:00");

        // Add reset button
        findViewById(R.id.reset_counter_button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                for (int i = 0; i < NUM_SCENARIOS; i++) {
                    countScenarios[i] = 0;
                    tvScenarios[i].setText("0");
                }
            }
        });
    }

    private synchronized void stopTensorFlow() {
        stopRecognition();
        stopRecording();
    }

    private synchronized void startTensorFlow() {

        // Load the labels for the model
        String actualFilename = LABEL_FILENAME.split("file:///android_asset/")[1];
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(getAssets().open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        // Set up an object to smooth recognition results to increase accuracy.
        recognizeCommands = new RecognizeCommands(
                labels,
                AVERAGE_WINDOW_DURATION_MS,
                DETECTION_THRESHOLD,
                SUPPRESSION_MS,
                MINIMUM_COUNT,
                MINIMUM_TIME_BETWEEN_SAMPLES_MS);

        // Load the TensorFlow model.
        inferenceInterface = new TensorFlowInferenceInterface(getAssets(), MODEL_FILENAME);

        // Start the recording and recognition threads.
        startRecording();
        startRecognition();
    }

    public synchronized void startRecording() {
        if (recordingThread != null) {
            return;
        }
        shouldContinue = true;
        recordingThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                record();
                            }
                        });
        recordingThread.start();

        handlerTimer = new Handler();
        runnableTimer = new Runnable() {
            @Override
            public void run() {
                timerCount++;
                timerDisplay.setText(String.format(Locale.ENGLISH,
                        "%02d:%02d", timerCount / 60, timerCount % 60));
                handlerTimer.postDelayed(runnableTimer, 1000);
            }
        };
        handlerTimer.post(runnableTimer);
    }

    public synchronized void stopRecording() {
        if (recordingThread == null) {
            return;
        }
        shouldContinue = false;
        recordingThread = null;

        timerCount = 0;
        timerDisplay.setText(String.format(Locale.ENGLISH, "%02d:%02d", 0, 0));
        handlerTimer.removeCallbacks(runnableTimer);
    }

    private void record() {
        android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO);

        // Estimate the buffer size we'll need for this device.
        int bufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (bufferSize == AudioRecord.ERROR || bufferSize == AudioRecord.ERROR_BAD_VALUE) {
            bufferSize = SAMPLE_RATE * 2;
        }
        short[] audioBuffer = new short[bufferSize / 2];

        AudioRecord record =
                new AudioRecord(
                        MediaRecorder.AudioSource.DEFAULT,
                        SAMPLE_RATE,
                        AudioFormat.CHANNEL_IN_MONO,
                        AudioFormat.ENCODING_PCM_16BIT,
                        bufferSize);

        if (record.getState() != AudioRecord.STATE_INITIALIZED) {
            Log.e(DBG_TAG, "Audio Record can't initialize!");
            return;
        }

        record.startRecording();

        Log.v(DBG_TAG, "Start recording");

        // Loop, gathering audio data and copying it to a round-robin buffer.
        while (shouldContinue) {
            int numberRead = record.read(audioBuffer, 0, audioBuffer.length);
            int maxLength = recordingBuffer.length;
            int newRecordingOffset = recordingOffset + numberRead;
            int secondCopyLength = Math.max(0, newRecordingOffset - maxLength);
            int firstCopyLength = numberRead - secondCopyLength;
            // We store off all the data for the recognition thread to access. The ML
            // thread will copy out of this buffer into its own, while holding the
            // lock, so this should be thread safe.
            recordingBufferLock.lock();
            try {
                System.arraycopy(audioBuffer, 0, recordingBuffer, recordingOffset, firstCopyLength);
                System.arraycopy(audioBuffer, firstCopyLength, recordingBuffer, 0, secondCopyLength);
                recordingOffset = newRecordingOffset % maxLength;
            } finally {
                recordingBufferLock.unlock();
            }
        }

        record.stop();
        record.release();
    }

    public synchronized void startRecognition() {
        if (recognitionThread != null) {
            return;
        }
        shouldContinueRecognition = true;
        recognitionThread =
                new Thread(
                        new Runnable() {
                            @Override
                            public void run() {
                                recognize();
                            }
                        });
        recognitionThread.start();

        setupVolumeHandler.post(setupVolumeRunnable);
        previousScenario = currentScenario = 0;
    }

    public synchronized void stopRecognition() {
        if (recognitionThread == null) {
            return;
        }
        shouldContinueRecognition = false;
        recognitionThread = null;

        fftCounter = 0;
        freqCounter = 0;
        freqInRangeCounter = 0;
        freqInRangeFlag = 2;

        setupVolumeHandler.removeCallbacks(setupVolumeRunnable);
        previousScenario = currentScenario = 0;
    }


    private void recognize() {
        Log.v(DBG_TAG, "Start recognition");

        short[] inputBuffer = new short[RECORDING_LENGTH];
        float[] floatInputBuffer = new float[RECORDING_LENGTH];
        float[] outputScores = new float[labels.size()];
        String[] outputScoresNames = new String[]{OUTPUT_SCORES_NAME};
        int[] sampleRateList = new int[]{SAMPLE_RATE};

        // Loop, grabbing recorded data and running the recognition model on it.
        while (shouldContinueRecognition) {
            // The recording thread places data in this round-robin buffer, so lock to
            // make sure there's no writing happening and then copy it to our own
            // local version.
            recordingBufferLock.lock();

            try {
                int maxLength = recordingBuffer.length;
                int firstCopyLength = maxLength - recordingOffset;
                int secondCopyLength = recordingOffset;
                System.arraycopy(recordingBuffer, recordingOffset, inputBuffer, 0, firstCopyLength);
                System.arraycopy(recordingBuffer, 0, inputBuffer, firstCopyLength, secondCopyLength);
            } finally {
                recordingBufferLock.unlock();
            }

            long currentTime = System.currentTimeMillis();

            // We need to feed in float values between -1.0f and 1.0f, so divide the
            // signed 16-bit inputs.
            double sum = 0, amplitude = 0;
            for (int i = 0; i < RECORDING_LENGTH; ++i) {
                floatInputBuffer[i] = inputBuffer[i] / 32767.0f;
                sum += floatInputBuffer[i] * floatInputBuffer[i];
            }

            // Get DB value from PCM encoding
            amplitude = 100 + 20 * Math.log10(Math.sqrt(sum / RECORDING_LENGTH) / 2);

            processLatestDbValues(currentTime, amplitude);

            fftCalculateFrequency(floatInputBuffer);


            // Run the model.
            inferenceInterface.feed(SAMPLE_RATE_NAME, sampleRateList);
            inferenceInterface.feed(INPUT_DATA_NAME, floatInputBuffer, RECORDING_LENGTH, 1);
            inferenceInterface.run(outputScoresNames);
            inferenceInterface.fetch(OUTPUT_SCORES_NAME, outputScores);

            // Use the smoother to figure out if we've had a real recognition event.
            final RecognizeCommands.RecognitionResult result =
                    recognizeCommands.processLatestResults(outputScores, currentTime);
//            Log.d(DBG_TAG, "isHumanVoiceDetected = " + result.isHumanVoiceDetected);

            runOnUiThread(
                    new Runnable() {
                        @Override
                        public void run() {

                            determineScenario(result);
                        }
                    });

            try {
                // We don't need to run too frequently, so snooze for a bit.
                Thread.sleep(MINIMUM_TIME_BETWEEN_SAMPLES_MS);

            } catch (InterruptedException e) {
                // Ignore
            }
        }

        Log.v(DBG_TAG, "End recognition");
    }

    private void fftCalculateFrequency(float[] floatInputBuffer) {
        // Do FFT
        fftCounter++;
        if (fftCounter > 25) {
            fftCounter = 0;

            Complex[] complexData = new Complex[FFT_INPUT_LENGTH];
            for (int i = 0; i < FFT_INPUT_LENGTH; i++) {
                complexData[i] = new Complex(floatInputBuffer[i], 0);
            }
            Complex[] fftResult = FFT.fft(complexData);
            Log.d(DBG_TAG, "fftResult length = " + fftResult.length);
            int maxFrequency = 0;
            for (int i = 0; i < fftResult.length; i++) {
                if (fftResult[i].abs() > fftResult[maxFrequency].abs()) {
                    maxFrequency = i;
                }
            }
            maxFrequency *= ((float)SAMPLE_RATE / FFT_INPUT_LENGTH);
            Log.d(DBG_TAG, "maxFrequency = " + maxFrequency + " HZ");

            freqCounter++;
            if (freqCounter == FREQ_MAX_COUNTER) {
                if (freqInRangeCounter > 4)
                    freqInRangeFlag = 2;
                else if (freqInRangeCounter >= 2)
                    freqInRangeFlag = 1;
                else
                    freqInRangeFlag = 0;
                freqCounter = 0;
                freqInRangeCounter = 0;
            }
            if (80 <= maxFrequency && maxFrequency <= 640) {
                freqInRangeCounter++;
            }
            Log.d(DBG_TAG, "freqInRangeCounter = " + freqInRangeCounter);
            Log.d(DBG_TAG, "freqInRangeFlag = " + freqInRangeFlag);
        }
    }

    private void determineScenario(RecognizeCommands.RecognitionResult result) {
//        Log.d(DBG_TAG, "result.foundCommand = " + result.foundCommand);

        // Scenario 1: "Silence" = Average db value < 40
        // Scenario 2: "Single voice" = majority db value > 70%
        // Scenario 4: "Ambiance Noise" = Average db value > 40 & no human voice detected
        // Scenario 3: "Crowd voice" = all other cases
        if (averageDb < DB_THRESHOLD_SILENCE) {

            Log.d(DBG_TAG, "Scenario 1: Silence");
            countScenarios[0]++;
            countScenariosLastOneMinute[0]++;
            tvScenarios[0].setText(String.format(Locale.ENGLISH, "%d", countScenarios[0]));

        } else if (!result.isHumanVoiceDetected) {

            Log.d(DBG_TAG, "Scenario 4: Ambiance Noise");
            countScenarios[3]++;
            countScenariosLastOneMinute[3]++;
            tvScenarios[3].setText(String.format(Locale.ENGLISH, "%d", countScenarios[3]));

        } else if (majorityDbRatio > SINGLE_VOICE_DB_RATIO &&
                averageDb < DB_THRESHOLD_SINGLE_VOICE) {

            if (freqInRangeFlag == 2) {
                Log.d(DBG_TAG, "Scenario 2: Single Voice");
                countScenarios[1]++;
                countScenariosLastOneMinute[1]++;
                tvScenarios[1].setText(String.format(Locale.ENGLISH, "%d", countScenarios[1]));
            } else if (freqInRangeFlag == 1) {
                Log.d(DBG_TAG, "Scenario 3: Crowd Voice");
                countScenarios[2]++;
                countScenariosLastOneMinute[2]++;
                tvScenarios[2].setText(String.format(Locale.ENGLISH, "%d", countScenarios[2]));
            } else {
                Log.d(DBG_TAG, "Scenario 4: Ambiance Noise");
                countScenarios[3]++;
                countScenariosLastOneMinute[3]++;
                tvScenarios[3].setText(String.format(Locale.ENGLISH, "%d", countScenarios[3]));
            }

        } else {

            Log.d(DBG_TAG, "Scenario 3: Crowd Voice");
            countScenarios[2]++;
            countScenariosLastOneMinute[2]++;
            tvScenarios[2].setText(String.format(Locale.ENGLISH, "%d", countScenarios[2]));

        }


        // If we do have a new command, highlight the right list entry.
        if (!result.foundCommand.startsWith("_") && result.isNewCommand) {
            Log.d(DBG_TAG, "result.foundCommand = " + result.foundCommand);
        }
    }

    private void processLatestDbValues(long currentTime, double amplitude) {

        if (amplitude < 0 || amplitude > 100) return;

        Double dbSum = averageDb * previousDbResults.size();

        // Add the latest results to the head of the queue.
        previousDbResults.addLast(new Pair<>(currentTime, amplitude));
        dbSum += amplitude;

        // Prune any earlier results that are too old for the averaging window.
        while (previousDbResults.getFirst().first < currentTime - DB_WINDOW_MS) {
            Pair<Long, Double> p = previousDbResults.removeFirst();
            dbSum -= p.second;
        }

        averageDb = dbSum / previousDbResults.size();
        Log.d(DBG_TAG, "averageDb = " + String.format("%3.2f", averageDb));

        majorityDbRatio = 0.0;
        for (Pair<Long, Double> db : previousDbResults) {
            if (averageDb - SINGLE_VOICE_DB_RANGE < db.second && db.second < averageDb + SINGLE_VOICE_DB_RANGE) {
                majorityDbRatio++;
            }
        }
        majorityDbRatio = majorityDbRatio / previousDbResults.size();
//        Log.d(DBG_TAG, "majorityDbRatio = " + String.format("%2.2f", majorityDbRatio * 100));
    }
}
