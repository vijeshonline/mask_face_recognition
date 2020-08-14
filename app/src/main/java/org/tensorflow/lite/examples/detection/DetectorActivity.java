/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.hardware.camera2.CameraCharacteristics;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import com.google.android.gms.tasks.OnFailureListener;
import com.google.android.gms.tasks.OnSuccessListener;
import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.firebase.storage.FirebaseStorage;
import com.google.firebase.storage.StorageReference;
import com.google.firebase.storage.UploadTask;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.Face;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectMaskDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();


  // FaceNet
//  private static final int TF_OD_API_INPUT_SIZE = 160;
//  private static final boolean TF_OD_API_IS_QUANTIZED = false;
//  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
//  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";

  // MobileFaceNet
  private static final int TF_OD_API_INPUT_SIZE = 112;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "mobile_face_net.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

  // Face Mask
  private static final int TF_OD_API_INPUT_SIZE_MASK = 224; //original was 224
  private static final boolean TF_OD_API_IS_QUANTIZED_MASK = false;
  private static final String TF_OD_API_MODEL_FILE_MASK = "mask_detector.tflite";
  private static final String TF_OD_API_LABELS_FILE_MASK = "file:///android_asset/mask_labelmap.txt";

//  private static final String RECOG_DIR_NAME = "FaceRecognitionFiles";

  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;

  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  //private static final int CROP_SIZE = 320;
  //private static final Size CROP_SIZE = new Size(320, 320);
  private static final String TAG = "MASK_FACE_RECOG";

  private static boolean emailSend = false;
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private SimilarityClassifier detector;
  private Classifier maskDetector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;
  private boolean addPending = false;
  //private boolean adding = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;
  //private Matrix cropToPortraitTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  // Face detector
  private FaceDetector faceDetector;

  // here the preview image is drawn in portrait way
  private Bitmap portraitBmp = null;
  // here the face is cropped and drawn
  private Bitmap faceBmp = null;
  private Bitmap maskFaceBmp = null;

  private FloatingActionButton fabAdd;
private String oldLabel = "";
  //private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();

  private SharedPreferences mPreference;
  private String sharedPrefFile = "org.tensorflow.lite.examples.detection.nameEmailPrefs";
  private StorageReference mStorageRef;

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    fabAdd = findViewById(R.id.fab_add);
    fabAdd.setOnClickListener(new View.OnClickListener() {
      @Override
      public void onClick(View view) {
        onAddClick();
      }
    });

    // Real-time contour detection of multiple faces
    FaceDetectorOptions options =
            new FaceDetectorOptions.Builder()
                    .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                    .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                    .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                    .build();


    FaceDetector detector = FaceDetection.getClient(options);

    faceDetector = detector;

    mPreference = getSharedPreferences(sharedPrefFile, MODE_PRIVATE);
    //checkWritePermission();
//    createDirForRecog(RECOG_DIR_NAME);
    mStorageRef = FirebaseStorage.getInstance().getReference();
  }



  private void onAddClick() {

    addPending = true;
    //Toast.makeText(this, "click", Toast.LENGTH_LONG ).show();

  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
            TypedValue.applyDimension(
                    TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);


    try {
      detector =
              TFLiteObjectDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE,
                      TF_OD_API_LABELS_FILE,
                      TF_OD_API_INPUT_SIZE,
                      TF_OD_API_IS_QUANTIZED);
      //cropSize = TF_OD_API_INPUT_SIZE;
      initSavedRecogFiles(detector);
//      initSavedRecogFilesFromAssets(detector);

    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    try {
      maskDetector =
              TFLiteObjectMaskDetectionAPIModel.create(
                      getAssets(),
                      TF_OD_API_MODEL_FILE_MASK,
                      TF_OD_API_LABELS_FILE_MASK,
                      TF_OD_API_INPUT_SIZE_MASK,
                      TF_OD_API_IS_QUANTIZED_MASK);
      //cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
              Toast.makeText(
                      getApplicationContext(), "MASK Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);


    int targetW, targetH;
    if (sensorOrientation == 90 || sensorOrientation == 270) {
      targetH = previewWidth;
      targetW = previewHeight;
    }
    else {
      targetW = previewWidth;
      targetH = previewHeight;
    }
    int cropW = (int) (targetW / 2.0);
    int cropH = (int) (targetH / 2.0);

    croppedBitmap = Bitmap.createBitmap(cropW, cropH, Config.ARGB_8888);

    portraitBmp = Bitmap.createBitmap(targetW, targetH, Config.ARGB_8888);
    faceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Config.ARGB_8888);
    maskFaceBmp = Bitmap.createBitmap(TF_OD_API_INPUT_SIZE_MASK, TF_OD_API_INPUT_SIZE_MASK, Config.ARGB_8888);

    frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    cropW, cropH,
                    sensorOrientation, MAINTAIN_ASPECT);

//    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);


    Matrix frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                    previewWidth, previewHeight,
                    targetW, targetH,
                    sensorOrientation, MAINTAIN_ASPECT);



    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
            new DrawCallback() {
              @Override
              public void drawCallback(final Canvas canvas) {
                tracker.draw(canvas);
                if (isDebug()) {
                  tracker.drawDebug(canvas);
                }
              }
            });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
  }


  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;

    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    InputImage image = InputImage.fromBitmap(croppedBitmap, 0);
    faceDetector
            .process(image)
            .addOnSuccessListener(new OnSuccessListener<List<Face>>() {
              @Override
              public void onSuccess(List<Face> faces) {
                if (faces.size() == 0) {
                  updateResults(currTimestamp, new LinkedList<>());
                  return;
                }
                runInBackground(
                        new Runnable() {
                          @Override
                          public void run() {
                            onFacesDetected(currTimestamp, faces, addPending);
                            addPending = false;
                          }
                        });
              }

            });


  }

  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }


  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
    runInBackground(() -> maskDetector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
    runInBackground(() -> maskDetector.setNumThreads(numThreads));
  }


  // Face Processing
  private Matrix createTransform(
          final int srcWidth,
          final int srcHeight,
          final int dstWidth,
          final int dstHeight,
          final int applyRotation) {

    Matrix matrix = new Matrix();
    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        LOGGER.w("Rotation of %d % 90 != 0", applyRotation);
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

//        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;

    if (applyRotation != 0) {

      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;

  }

  private void showAddFaceDialog(SimilarityClassifier.Recognition rec) {

    AlertDialog.Builder builder = new AlertDialog.Builder(this);
    LayoutInflater inflater = getLayoutInflater();
    View dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null);
    ImageView ivFace = dialogLayout.findViewById(R.id.dlg_image);
    TextView tvTitle = dialogLayout.findViewById(R.id.dlg_title);
    EditText etName = dialogLayout.findViewById(R.id.dlg_input);
    EditText toEmail = dialogLayout.findViewById(R.id.to_email);

    tvTitle.setText("Add Face");
    ivFace.setImageBitmap(rec.getCrop());
    etName.setHint("Input name");
    toEmail.setHint("Enter Email");

    builder.setPositiveButton("OK", new DialogInterface.OnClickListener(){
      @Override
      public void onClick(DialogInterface dlg, int i) {

          String name = etName.getText().toString();
          if (name.isEmpty()) {
            Toast.makeText(getApplicationContext(),"Name and email is required !!!", Toast.LENGTH_LONG).show();
            return;
          }

          //
          String email = toEmail.getText().toString();
          Log.i(TAG,email);
          boolean emailStatus =  android.util.Patterns.EMAIL_ADDRESS.matcher(email).matches();
          if (emailStatus){
            SharedPreferences.Editor prefEditor = mPreference.edit();
            prefEditor.putString(name,email);
            prefEditor.apply();
            Log.i(TAG, "Email added to sharedPref: "+email);
          }else{
            Log.i(TAG,"Wrong email!!!");
            Toast.makeText(getApplicationContext(),"Wrong email format !!!", Toast.LENGTH_SHORT).show();
            return;
          }
          writeRecordsToFile(name, rec);
          writeRecordsToExternalDir(name, rec);

          if(rec != null){
            detector.register(name, rec);
          }else{
            Log.i(TAG,"Failed to save face recognition data....");
          }
          //TODO remove till this

//          detector.register(name, rec); //TODO to be uncommented.
          //knownFaces.put(name, rec);
          dlg.dismiss();
      }
    });
    builder.setView(dialogLayout);
    builder.show();

  }

  private void updateResults(long currTimestamp, final List<SimilarityClassifier.Recognition> mappedRecognitions) {

    tracker.trackResults(mappedRecognitions, currTimestamp);
    trackingOverlay.postInvalidate();
    computingDetection = false;
    //adding = false;


    if (mappedRecognitions.size() > 0) {
       LOGGER.i("Adding results");
       SimilarityClassifier.Recognition rec = mappedRecognitions.get(0);
       if (rec.getExtra() != null) {
         showAddFaceDialog(rec);
       }

    }

    runOnUiThread(
            new Runnable() {
              @Override
              public void run() {
                showFrameInfo(previewWidth + "x" + previewHeight);
                showCropInfo(croppedBitmap.getWidth() + "x" + croppedBitmap.getHeight());
                showInference(lastProcessingTimeMs + "ms");
              }
            });

  }

  private void onFacesDetected(long currTimestamp, List<Face> faces, boolean add) {

    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
    final Canvas canvas = new Canvas(cropCopyBitmap);
    final Paint paint = new Paint();
    paint.setColor(Color.RED);
    paint.setStyle(Style.STROKE);
    paint.setStrokeWidth(2.0f);
    Log.v(TAG,"onFaceDetected");
    float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
    switch (MODE) {
      case TF_OD_API:
        minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
        break;
    }

    final List<SimilarityClassifier.Recognition> mappedRecognitions =
            new LinkedList<SimilarityClassifier.Recognition>();


    //final List<Classifier.Recognition> results = new ArrayList<>();

    // Note this can be done only once
    int sourceW = rgbFrameBitmap.getWidth();
    int sourceH = rgbFrameBitmap.getHeight();
    int targetW = portraitBmp.getWidth();
    int targetH = portraitBmp.getHeight();
    Matrix transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation);
    final Canvas cv = new Canvas(portraitBmp);

    // draws the original image in portrait mode.
    cv.drawBitmap(rgbFrameBitmap, transform, null);

    final Canvas cvFace = new Canvas(faceBmp);

    boolean saved = false;

    for (Face face : faces) {

      LOGGER.i("FACE" + face.toString());
      LOGGER.i("Running detection on face " + currTimestamp);
      //results = detector.recognizeImage(croppedBitmap);

      final RectF boundingBox = new RectF(face.getBoundingBox());

      //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
      final boolean goodConfidence = true; //face.get;
      if (boundingBox != null && goodConfidence) {

        // maps crop coordinates to original
        cropToFrameTransform.mapRect(boundingBox);

        // maps original coordinates to portrait coordinates
        RectF faceBB = new RectF(boundingBox);
        transform.mapRect(faceBB);

        // translates portrait to origin and scales to fit input inference size
        //cv.drawRect(faceBB, paint);
        float sx = ((float) TF_OD_API_INPUT_SIZE) / faceBB.width();
        float sy = ((float) TF_OD_API_INPUT_SIZE) / faceBB.height();
        Matrix matrix = new Matrix();
        matrix.postTranslate(-faceBB.left, -faceBB.top);
        matrix.postScale(sx, sy);

        cvFace.drawBitmap(portraitBmp, matrix, null);

        //
        final Canvas maskcvFace = new Canvas(maskFaceBmp);
        float sx2 = ((float) TF_OD_API_INPUT_SIZE_MASK) / faceBB.width();
        float sy2 = ((float) TF_OD_API_INPUT_SIZE_MASK) / faceBB.height();
        Matrix matrix2 = new Matrix();
        matrix2.postTranslate(-faceBB.left, -faceBB.top);
        matrix2.postScale(sx2, sy2);
        maskcvFace.drawBitmap(portraitBmp,matrix2,null);
        boolean masked = isMaskDetected(maskFaceBmp);
        //

        //canvas.drawRect(faceBB, paint);

        String label = "";
        float confidence = -1f;
        Integer color = Color.BLUE;
        Object extra = null;
        Bitmap crop = null;

        if (add) {
          crop = Bitmap.createBitmap(portraitBmp,
                            (int) faceBB.left,
                            (int) faceBB.top,
                            (int) faceBB.width(),
                            (int) faceBB.height());
        }

//        if(masked != true) { //if no mask detected, upload bmp for face recgonition.
//          uploadBitMaptoFirebase(portraitBmp);
//        }

        final long startTime = SystemClock.uptimeMillis();
        final List<SimilarityClassifier.Recognition> resultsAux = detector.recognizeImage(faceBmp, add);
        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

        //
        color = (masked)?Color.GREEN : Color.RED;
        //

        if (resultsAux.size() > 0) {

          SimilarityClassifier.Recognition result = resultsAux.get(0);

          extra = result.getExtra();
//          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }
          label = (masked)?"Masked":"Unknown Face";
          float conf = result.getDistance();
          if (conf < 1.0f) {

            confidence = conf;
            label = result.getTitle();
            if (result.getId().equals("0")) {
              //color = Color.GREEN;
              color = (masked)?Color.GREEN : Color.RED;
              Log.i(TAG, oldLabel + "------" + label);
              if(label != oldLabel) {
                Toast.makeText(this.getApplicationContext(), "Warning email sent to " + label, Toast.LENGTH_SHORT).show();
              }
              oldLabel = label;
            } else {
              //color = Color.RED;
              label = "Unknown";
              color = (masked)?Color.GREEN : Color.RED;
            }
          }
        }

        if (getCameraFacing() == CameraCharacteristics.LENS_FACING_FRONT) {

          // camera is frontal so the image is flipped horizontally
          // flips horizontally
          Matrix flip = new Matrix();
          if (sensorOrientation == 90 || sensorOrientation == 270) {
            flip.postScale(1, -1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          else {
            flip.postScale(-1, 1, previewWidth / 2.0f, previewHeight / 2.0f);
          }
          //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
          flip.mapRect(boundingBox);

        }

        final SimilarityClassifier.Recognition result = new SimilarityClassifier.Recognition(
                "0", label, confidence, boundingBox);

        result.setColor(color);
        result.setLocation(boundingBox);
        result.setExtra(extra);
        result.setCrop(crop);
        mappedRecognitions.add(result);

        //String email = org.tensorflow.lite.examples.detection.Config.EMAIL;
        if (color == Color.GREEN && !emailSend) {
          String email = "projecttesting535@gmail.com";
          String subject = label + " is not wearing mask";
          String body = "test mail";
          emailSend = true;
          email = mPreference.getString(label, email);
          SendMail sendMail = new SendMail(this, email, subject, body);
          //sendMail.execute(); //TODO this should be enabled to send mail. GMAIL blocks it often.
          Log.i(TAG, "Mail send to " + email);
        }else{
//          Log.i(TAG, label + " Unknown face");
        }
      }


    }

    //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }

    updateResults(currTimestamp, mappedRecognitions);


  }

  private boolean isMaskDetected(Bitmap localFaceBmp) {
    boolean maskedFace = false;
    String label = "";
    float confidence = -1f;
    Integer color = Color.BLUE;

    //final long startTime = SystemClock.uptimeMillis();
    final List<Classifier.Recognition> resultsAux = maskDetector.recognizeImage(localFaceBmp);
    //lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

    if (resultsAux.size() > 0) {
      Classifier.Recognition result = resultsAux.get(0);
      float conf = result.getConfidence();
      if (conf >= 0.6f) {
        confidence = conf;
        label = result.getTitle();
        Log.d(TAG,"isMaskDetected: LABEL: " + label +", Confidence: "+confidence);
        if (result.getId().equals("0")) {
          Log.i (TAG,"isMaskDetected: MASK   -------------------------------------");
          maskedFace = true;
        }
        else {
          Log.i (TAG,"NO MASK   ???????????????????????????????????");
//          if ( confidence > 0.9999f){
//            ringAlarmSound();
//          }
        }
      }

    }else{
      Log.i(TAG, "No result from mask detection" );

    }

    return maskedFace;
  }

  public boolean writeRecordsToExternalDir(String fileName, SimilarityClassifier.Recognition recog){

    ObjectOutputStream oos=null;
//    String filePath = RECOG_DIR_NAME+"/"+fileName;

    FileOutputStream fos = null;
    try{
      File root = Environment.getExternalStorageDirectory();
      File dir = new File (root.getAbsolutePath() + "/download");
      dir.mkdirs();
      File file = new File(dir, fileName);
      fos = new FileOutputStream(file);
      //fos = getApplicationContext().openFileOutput(fileName, Context.MODE_PRIVATE);
      oos = new ObjectOutputStream(fos);
      oos.writeObject(recog);
      oos.close();
      Log.v(TAG, "WriteRecordsToExtFile: Recog Object: " + dir.getPath()+ " recog:: " + recog.toString());
      fos.close();
      return true;
    }catch(Exception e){
      Log.e(TAG, "Cant save records:  "+e.getMessage());
      return false;
    }
    finally{
      if(oos!=null && fos != null)
        try{
          oos.close();
          fos.close();
        }catch(Exception e){
          Log.e(TAG, "Error while closing stream "+e.getMessage());
        }
    }
  }
/*
* Write face data to external storage.
* This data will be used to teach the face to system, when APK is launched every time.
* */

  public boolean writeRecordsToFile(String fileName, SimilarityClassifier.Recognition recog){

    ObjectOutputStream oos=null;
//    String filePath = RECOG_DIR_NAME+"/"+fileName;
    FileOutputStream fos;
    try{
      //fos = new FileOutputStream(fileName);
      fos = getApplicationContext().openFileOutput(fileName, Context.MODE_PRIVATE);
      oos = new ObjectOutputStream(fos);
      oos.writeObject(recog);
      oos.close();
      Log.v(TAG, "WriteRecordsToFile: Recog Object: " + recog.toString());
      fos.close();
      return true;
    }catch(Exception e){
      Log.e(TAG, "Cant save records:  "+e.getMessage());
      return false;
    }
    finally{
      if(oos!=null)
        try{
          oos.close();
        }catch(Exception e){
          Log.e(TAG, "Error while closing stream "+e.getMessage());
        }
    }
  }

  private SimilarityClassifier.Recognition readRecordsFromFile(String fileName){
    FileInputStream fin;
    ObjectInputStream ois=null;
//    String filePath = RECOG_DIR_NAME+"/"+fileName;
    try{
      fin = getApplicationContext().openFileInput(fileName);
      ois = new ObjectInputStream(fin);
      SimilarityClassifier.Recognition recog = (SimilarityClassifier.Recognition) ois.readObject();
      ois.close();
      Log.v(TAG, "ReadRecordsFromFile: Recog Object: " + recog.toString());
      return recog;
    }catch(Exception e){
      Log.e(TAG, "Cant read saved records   "+e.getMessage());
      return null;
    }
    finally{
      if(ois!=null)
        try{
          ois.close();
        }catch(Exception e){
          Log.i(TAG, "Error in closing stream while reading records"+e.getMessage());
        }
    }
  }
/*
* Read JPEG files (saved as .mp3) from assets folder and return Recognition object.
* */
  private SimilarityClassifier.Recognition readRecordsFromAssets(String fileName){
    FileInputStream fin;
    ObjectInputStream ois=null;

    try{
      fin = getAssets().openFd(fileName).createInputStream();
      ois = new ObjectInputStream(fin);
      SimilarityClassifier.Recognition recog = (SimilarityClassifier.Recognition) ois.readObject();
      ois.close();
      Log.v(TAG, "readRecordsFromAssets: Recog Object: " + recog.toString());
      return recog;
    }catch(Exception e){
      Log.e(TAG, "readRecordsFromAssets Cant read saved records "+fileName + e.getMessage());
      return null;
    }

    finally{
      if(ois!=null)
        try{
          ois.close();
        }catch(Exception e){
          Log.i(TAG, "Error in closing stream while reading records"+e.getMessage());
        }
    }
  }
/*
Face data is stored in external director folder.
Read from ext dir and set recog data to detector.
Face data can be saved in external directory. To avoid teaching every face manually.
.MP3 extension added to avoid compression by Android. (.JPG will get compressed and read fails)
*/
  private void initSavedRecogFiles(SimilarityClassifier detector) {
    try {
      //File folder = Environment.getDataDirectory();
      File folder = getApplicationContext().getFilesDir();
      File[] listOfFiles = folder.listFiles();

      for (File file : listOfFiles) {
        if (file.isFile()) {
          String fileName = file.getName();
          SimilarityClassifier.Recognition recog = readRecordsFromFile(fileName);
          if (null != recog) {
            detector.register(fileName, recog); //label is same as filename
            Log.v(TAG, "Initializing recognition data file " + folder.getPath() + fileName);
          }
        }
      }
    }catch (Exception e){
      e.printStackTrace();
      Log.i(TAG, e.getMessage());
    }
  }

//  /*
//  Face data is stored in assets folder.
//  Read from assets and set recog data to detector.
//  This will help to deliver APK with pre-trained faces. (APK size will increase.
//  .MP3 extension added to avoid compression by Android. (.JPG will get compressed and read fails)
//  */
//  private void initSavedRecogFilesFromAssets(SimilarityClassifier detector) {
//    try {
//      String[] fileNameList = {"v5.mp3", "anees.mp3", "aneesU.mp3"};
//
//      for (String fileName : fileNameList) {
//        SimilarityClassifier.Recognition recog = readRecordsFromAssets(fileName);
//        if (null != recog) {
//          detector.register(stripExtension(fileName), recog); //label is same as filename
//          Log.v(TAG, "initSavedRecogFilesFromAssets: Initializing recognition data file From Assets: " + fileName);
//        }
//      }
//    }catch(Exception e){
//        e.printStackTrace();
//        Log.i(TAG, e.getMessage());
//    }
//  }


  private String stripExtension(String fileName) {
    if (fileName.indexOf(".") > 0) {
      return fileName.substring(0, fileName.lastIndexOf("."));
    } else {
      return fileName;
    }
  }
  private void createDirForRecog(String dirName) {
    File mediaStorageDir = new File(Environment.getExternalStorageDirectory(), dirName);

    if (!mediaStorageDir.exists()) {
      if (!mediaStorageDir.mkdirs()) {
        Log.d("App", "failed to create directory");
      }
    }
  }

//  private String saveBitmapToLocalStorage(Bitmap faceBmp){
//    String jpegPath = "path";
//    try {
//      String root = Environment.getExternalStorageDirectory().toString();
//      File myDir = new File(root + "/req_images");
//      myDir.mkdirs();
//      Random generator = new Random();
//      int n = 10000;
//      n = generator.nextInt(n);
//      String fname = "Image-" + n + ".jpg";
//      File file = new File(myDir, fname);
//      Log.i(TAG, "" + file);
//      if (file.exists())
//        file.delete();
//      try {
//        FileOutputStream out = new FileOutputStream(file);
//        faceBmp.compress(Bitmap.CompressFormat.JPEG, 90, out);
//        out.flush();
//        out.close();
//      } catch (Exception e) {
//        e.printStackTrace();
//      }
//
//      jpegPath = root + "/req_image/" + fname;
//      Log.i(TAG,"JPEG save to path: " + jpegPath);
//    }catch (Exception e){
//      e.printStackTrace();
//      Log.e(TAG, "Failed to write file to memory");
//    }
//    return jpegPath;
//  }
  private void uploadBitMaptoFirebase(Bitmap faceBmp){
    Log.i(TAG, "Uploading files to firebase cloud.........");
    Random generator = new Random(); //create a random file name
    int n = 10000;
    n = generator.nextInt(n);
    String fname = "Image-50-" + n + ".jpg";

    StorageReference faceRef = mStorageRef.child("ForRecognition/"+fname);

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    faceBmp.compress(Bitmap.CompressFormat.JPEG, 100, baos);
    byte[] data = baos.toByteArray();

    UploadTask uploadTask = faceRef.putBytes(data);
    uploadTask.addOnFailureListener(new OnFailureListener() {
      @Override
      public void onFailure(@NonNull Exception exception) {
        // Handle unsuccessful uploads
      }
    }).addOnSuccessListener(new OnSuccessListener<UploadTask.TaskSnapshot>() {
      @Override
      public void onSuccess(UploadTask.TaskSnapshot taskSnapshot) {
        // taskSnapshot.getMetadata() contains file metadata such as size, content-type, etc.
        // ...
        Log.i(TAG,"uploaded: "+taskSnapshot.toString());
      }
    });


  }
}