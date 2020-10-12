package com.example.myshite.App;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Matrix;
import android.media.ExifInterface;
import android.os.Bundle;
import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import com.example.myshite.App.Database.SQLiteHelper;
import com.example.myshite.R;

import com.github.clans.fab.FloatingActionButton;
import com.github.clans.fab.FloatingActionMenu;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import static org.opencv.core.Core.NORM_HAMMING;
import static org.opencv.core.Core.NORM_HAMMING2;
import static org.opencv.core.Core.norm;
import static org.opencv.core.CvType.CV_32F;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_RGB2GRAY;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class StartActivity extends AppCompatActivity {
    private Mat img_output;
    private Mat img_input;
    private Mat descriptor;
    ImageView imageVIewOuput;
    public static SQLiteHelper sqLiteHelper;
    ImageView viewImage;
    private static final String FILE_NAME = "example.txt";
    Button b;
    FloatingActionMenu materialDesignFAM;
    String path ="/mnt/sdcard/hello.txt";
    FloatingActionButton floatingActionButton1, floatingActionButton2, floatingActionButton3;
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_start);
        b=(Button)findViewById(R.id.btnSelectPhoto);
        viewImage=(ImageView)findViewById(R.id.viewImage);
        imageVIewOuput = (ImageView) findViewById(R.id.imageViewOutput);
        b.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectImage();
            }
        });
        materialDesignFAM = (FloatingActionMenu) findViewById(R.id.material_design_android_floating_action_menu);
        floatingActionButton1 = (FloatingActionButton) findViewById(R.id.material_design_floating_action_menu_item1);
        floatingActionButton2 = (FloatingActionButton) findViewById(R.id.material_design_floating_action_menu_item2);
        floatingActionButton3 = (FloatingActionButton) findViewById(R.id.material_design_floating_action_menu_item3);
        sqLiteHelper = new SQLiteHelper(this, "ObjectDB.sqlite", null, 1);

        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS ORBFEATURE(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");
        //sqLiteHelper.queryData("select  features from OBJECTFEATURE ");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS LBP(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");

        floatingActionButton1.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //TODO orb
                orbdescriptor();

            }
        });
        floatingActionButton2.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //TODO Lbp
                lbpedescriptore();
            }
        });
        floatingActionButton3.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                //TODO canny edge
                  Canny();
            }
        });
    }
public native void Orbcal(long inputImage, long outputImage);

    private void orbdescriptor() {

      //  descriptor = new Mat();
        if (img_output == null)
            img_output = new Mat();
          Orbcal(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        /*   int [] arr =new int[(int) (img_output.total() * img_output.channels())];
          double probablitit =0;

        Mat bb= new Mat();
                bb=sqLiteHelper.dbget("apple01");
        Toast.makeText(getApplicationContext(), "Get it!", Toast.LENGTH_SHORT).show();
            Cursor cursor = this.sqLiteHelper.getData("SELECT features FROM OBJECTFEATURE");
       int[] ret = new int[arr.length];
        int i = 0;
        while(cursor .moveToNext()){
            int  uname = cursor.getInt(cursor.getColumnIndex("features"));
            ret [i] = uname;
            i++;
        }


        probablitit = compare(ret ,arr);
*/
      /*  int size = s.length();
        int [] arr = new int [size];
        for(int i=0; i<size; i++) {
            arr[i] = Integer.parseInt(s);
        }
        System.out.println(Arrays.toString(arr));
        try{
            sqLiteHelper.insertData1(
                    "apple01".trim(),
                    arr
            );

        }
        catch (Exception e){
            e.printStackTrace();
        }*/

           // l2 for surf,sift
       /* try{
            sqLiteHelper.dbput1(
                    "apple02".trim(),
                    img_output
            );
            Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
        }
        catch (Exception e){
            e.printStackTrace();
        }*/
       Mat bb= sqLiteHelper.dbget1("apple02");
        Bitmap bitmapOutput = Bitmap.createBitmap(img_output.cols(), img_output.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_output, bitmapOutput);
        imageVIewOuput.setImageBitmap(bitmapOutput);
       cvtColor(img_output, img_output, COLOR_RGB2GRAY);
        cvtColor(bb, bb, COLOR_RGB2GRAY);
        double dist_ham = Core.norm(img_output,img_output,NORM_HAMMING2);

        //double ssim = CompareUtil.compareSSIM( img_output,bb);
        Toast.makeText(getApplicationContext(), "the probbilité!"+dist_ham , Toast.LENGTH_SHORT).show();
    }
    public static double compare(int [] f1, int []  f2) {
        if (f1.length == 0 || f2.length == 0 || f1.length != f2.length)
            return 0;
        double probability = 0;

        for (int i = 0; i < f1.length; i++) {
            if (f1[i] == f2[i]) {
                probability++;
            }
        }

       // probability /= f1.length;
        return probability;
    }


    public native String LbpCal(long inputImage, long outputImage,String path);
    
    private void lbpedescriptore() {
        if (img_output == null)
            img_output = new Mat();

        String msg= LbpCal(img_input.getNativeObjAddr(), img_output.getNativeObjAddr(),path);
        /*
        try{
            sqLiteHelper.dbput(
                    "apple01".trim(),
                    img_output
            );
            Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
        }
        catch (Exception e){
            e.printStackTrace();
        }*/
        Mat bb= sqLiteHelper.dbget("apple01");
        Toast.makeText(getApplicationContext(), "Get it!", Toast.LENGTH_SHORT).show();
        FileOutputStream fos = null;
        try {
            fos = openFileOutput(FILE_NAME, MODE_PRIVATE);
            fos.write(msg.getBytes());


            Toast.makeText(this, "Saved to " + getFilesDir() + "/" + FILE_NAME,
                    Toast.LENGTH_LONG).show();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (fos != null) {
                try {
                    fos.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }

        Bitmap bitmapOutput = Bitmap.createBitmap(img_output.cols(), img_output.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_output, bitmapOutput);
        imageVIewOuput.setImageBitmap(bitmapOutput);
        img_output.convertTo(img_output, CV_32F);
        bb.convertTo(bb, CV_32F);
////normalize the two historamme
        double res = Imgproc.compareHist(img_output, img_output, Imgproc.CV_COMP_CHISQR);
        Double d = new Double(res * 100);

        Toast.makeText(getApplicationContext(), "probbilite !"+d, Toast.LENGTH_SHORT).show();
    }

    public native void DetectEdge(long inputImage, long outputImage);
    private void Canny() {
        if (img_output == null)
            img_output = new Mat();

       DetectEdge(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        Bitmap bitmapOutput = Bitmap.createBitmap(img_output.cols(), img_output.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(img_output, bitmapOutput);
        imageVIewOuput.setImageBitmap(bitmapOutput);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds options to the action bar if it is present.
        getMenuInflater().inflate(R.menu.main, menu);
        return true;
    }
    private void selectImage() {
        final CharSequence[] options = { "Take Photo", "Choose from Gallery","Cancel" };
        AlertDialog.Builder builder = new AlertDialog.Builder(StartActivity.this);
        builder.setTitle("Add Photo!");
        builder.setItems(options, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int item) {
                if (options[item].equals("Take Photo"))
                {
                    Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    File f = new File(android.os.Environment.getExternalStorageDirectory(), "temp.jpg");
                    intent.putExtra(MediaStore.EXTRA_OUTPUT, Uri.fromFile(f));
                    startActivityForResult(intent, 1);
                }
                else if (options[item].equals("Choose from Gallery"))
                {
                    Intent intent = new   Intent(Intent.ACTION_PICK,android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                    startActivityForResult(intent, 2);

                }
                else if (options[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK) {
            if (requestCode == 1) {
                File f = new File(Environment.getExternalStorageDirectory().toString());
                for (File temp : f.listFiles()) {
                    if (temp.getName().equals("temp.jpg")) {
                        f = temp;
                        break;
                    }
                }
                try {
                    Bitmap bitmap;
                    BitmapFactory.Options bitmapOptions = new BitmapFactory.Options();
                    bitmap = BitmapFactory.decodeFile(f.getAbsolutePath(),
                            bitmapOptions);
                    viewImage.setImageBitmap(bitmap);
                    img_input = new Mat();
                    Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Utils.bitmapToMat(bmp32, img_input);
                    String path = android.os.Environment
                            .getExternalStorageDirectory()
                            + File.separator
                            + "Phoenix" + File.separator + "default";
                    f.delete();
                    OutputStream outFile = null;
                    File file = new File(path, String.valueOf(System.currentTimeMillis()) + ".jpg");
                    try {
                        outFile = new FileOutputStream(file);
                        bitmap.compress(Bitmap.CompressFormat.PNG, 85, outFile);

                        outFile.flush();
                        outFile.close();

                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else if (requestCode == 2) {
                Uri selectedImage = data.getData();
                String path = getRealPathFromURI(selectedImage);
                int orientation = getOrientationOfImage(path); // 런타임 퍼미션 필요
                Bitmap temp = null;
                try {
                    temp = MediaStore.Images.Media.getBitmap(getContentResolver(), selectedImage);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                Bitmap bitmap = null;
                try {
                    bitmap = getRotatedBitmap(temp, orientation);
                } catch (Exception e) {
                    e.printStackTrace();
                }
                viewImage.setImageBitmap(bitmap);

                img_input = new Mat();
                Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                Utils.bitmapToMat(bmp32, img_input);
                final Size kernelSize = new Size(11, 11);
                final Point anchor = new Point(-1, -1);
                final int iterations = 2;

                Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, kernelSize);
                Imgproc.erode(img_input, img_input, kernel, anchor, iterations);
                Imgproc.dilate(img_input, img_input, kernel, anchor, iterations);
            }
        }
    }


    private String getRealPathFromURI(Uri contentUri) {

        String[] proj = {MediaStore.Images.Media.DATA};
        Cursor cursor = getContentResolver().query(contentUri, proj, null, null, null);
        cursor.moveToFirst();
        int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);

        return cursor.getString(column_index);
    }

    // 출처 - http://snowdeer.github.io/android/2016/02/02/android-image-rotation/
    public int getOrientationOfImage(String filepath) {
        ExifInterface exif = null;

        try {
            exif = new ExifInterface(filepath);
        } catch (IOException e) {
            Log.d("@@@", e.toString());
            return -1;
        }

        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, -1);

        if (orientation != -1) {
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return 90;

                case ExifInterface.ORIENTATION_ROTATE_180:
                    return 180;

                case ExifInterface.ORIENTATION_ROTATE_270:
                    return 270;
            }
        }

        return 0;
    }

    public Bitmap getRotatedBitmap(Bitmap bitmap, int degrees) throws Exception {
        if(bitmap == null) return null;
        if (degrees == 0) return bitmap;

        Matrix m = new Matrix();
        m.setRotate(degrees, (float) bitmap.getWidth() / 2, (float) bitmap.getHeight() / 2);

        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), m, true);
    }



}

