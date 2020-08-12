/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.detection.tflite;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.RectF;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.List;

/** Generic interface for interacting with different recognition engines. */
public interface SimilarityClassifier {

  void register(String name, Recognition recognition);

  List<Recognition> recognizeImage(Bitmap bitmap, boolean getExtra);

  void enableStatLogging(final boolean debug);

  String getStatString();

  void close();

  void setNumThreads(int num_threads);

  void setUseNNAPI(boolean isChecked);

  /** An immutable result returned by a Classifier describing what was recognized. */
  public class Recognition implements Serializable{
    private static final long serialVersionUID = 6529685098267757690L;
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /** Display name for the recognition. */
    private final String title;

    /**
     * A sortable score for how good the recognition is relative to others. Lower should be better.
     */
    private final Float distance;
    private Object extra;

    /** Optional location within the source image for the location of the recognized object. */
    private transient RectF location;
    private Integer color;
    private transient Bitmap crop;

    public Recognition (
            final String id, final String title, final Float distance, final RectF location) {
      this.id = id;
      this.title = title;
      this.distance = distance;
      this.location = location;
      this.color = null;
      this.extra = null;
      this.crop = null;
    }

    public void setExtra(Object extra) {
        this.extra = extra;
    }
    public Object getExtra() {
        return this.extra;
    }

    public void setColor(Integer color) {
       this.color = color;
    }

    public String getId() {
      return id;
    }

    public String getTitle() {
      return title;
    }

    public Float getDistance() {
      return distance;
    }

    public RectF getLocation() {
      return new RectF(location);
    }

    public void setLocation(RectF location) {
      this.location = location;
    }

    @Override
    public String toString() {
      String resultString = "";
      if (id != null) {
        resultString += "[" + id + "] ";
      }

      if (title != null) {
        resultString += title + " ";
      }

      if (distance != null) {
        resultString += String.format("(%.1f%%) ", distance * 100.0f);
      }

      if (location != null) {
        resultString += location + " ";
      }

      return resultString.trim();
    }

    public Integer getColor() {
      return this.color;
    }

    public void setCrop(Bitmap crop) {
      this.crop = crop;
    }

    public Bitmap getCrop() {
      return this.crop;
    }

    //VIJESH

    private void writeObject(ObjectOutputStream oos) throws IOException {
      // This will serialize all fields that you did not mark with 'transient'
      // (Java's default behaviour)
      oos.defaultWriteObject();

      Float top = location.top;
      Float bottom = location.bottom;
      Float left = location.left;
      Float right = location.right;
      oos.writeFloat(top);
      oos.writeFloat(bottom);
      oos.writeFloat(left);
      oos.writeFloat(right);

      // Now, manually serialize all transient fields that you want to be serialized
      if(crop!=null){
        ByteArrayOutputStream byteStream = new ByteArrayOutputStream();
        boolean success = crop.compress(Bitmap.CompressFormat.PNG, 100, byteStream);
        if(success){
          oos.writeObject(byteStream.toByteArray());
        }
      }
    }

    private void readObject(ObjectInputStream ois) throws IOException, ClassNotFoundException{
      // Now, all again, deserializing - in the SAME ORDER!
      // All non-transient fields
      ois.defaultReadObject();
      this.location = new RectF();
      this.location.top = ois.readFloat();
      this.location.bottom = ois.readFloat();
      this.location.left = ois.readFloat();
      this.location.right = ois.readFloat();
      // All other fields that you serialized
      byte[] image = (byte[]) ois.readObject();
      if(image != null && image.length > 0){
        crop = BitmapFactory.decodeByteArray(image, 0, image.length);
      }
    }
    //VIJESH
  }
}
