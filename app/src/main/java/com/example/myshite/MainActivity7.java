package com.example.myshite;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import com.andremion.floatingnavigationview.FloatingNavigationView;
import com.example.myshite.App.Activitycheby;
import com.example.myshite.App.Database.SQLiteHelper;
import com.google.android.material.navigation.NavigationView;

import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import static org.opencv.core.CvType.CV_32F;
import static org.opencv.core.CvType.CV_8U;


class llbp {
    public double distance;
    public String namobject;
    public llbp(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}

class couleur{
    public double distance;
    public String namobject;
    public couleur(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}

class shapeu {
    public double distance;
    public String namobject;
    public shapeu(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class chebyy {
    public double distance;
    public String namobject;
    public chebyy(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class hu {
    public double distance;
    public String namobject;
    public hu(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class huu {
    public double distance;
    public String namobject;
    public huu(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class distancepobdre {
    public double distance;
    public String namobject;
    public distancepobdre(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class Sortbyroll4 implements Comparator<distancepobdre>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(distancepobdre a, distancepobdre b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll implements Comparator<chebyy>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(chebyy a, chebyy b)
    {
        return Double.compare(a.distance, b.distance);
    }
}class Sortbyrollhu implements Comparator<hu>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(hu a, hu b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyrollhuu implements Comparator<huu>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(huu a, huu b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll1 implements Comparator<llbp>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(llbp a, llbp b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll2 implements Comparator<couleur>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(couleur a, couleur b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll3 implements Comparator<shapeu>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(shapeu a, shapeu b)
    {
        return Double.compare(a.distance, b.distance);
    }
}

public class MainActivity7 extends AppCompatActivity {
    ImageView imag, imageVIewOuput;
    private Mat img_output, out, filtinput, canyinput, orbinput, lbpinput;
    private Mat img_input, matdegris, matdecanny, matdeorb, matdelbp, matfilter;
    Uri mImageUri;
    Bitmap bitmap; byte[] byteArray;
    Uri u ;
    boolean clicked = false;
    Context context;
    boolean clikedgris = false;
    boolean clikedgaussien = false;
    boolean clikedcanny = false;
    int i = 0;
    double w3=0.6355;//couleur
    double w1=0.3052;//chebychev
    double w2=0.5842;//elbp
    Thread t=null;
    private int camera = 1,gellary = 2;

    public static final int PERMISSION_CODE = 111;
    Thread t1=null;
    Thread t2=null;
    String path = "/mnt/sdcard/lbp/example.txt";
    TextView textView, textView1;
    public static SQLiteHelper sqLiteHelper;
    ProgressBar Progress;
    private FloatingNavigationView mFloatingNavigationView;

    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }
    List<couleur> distancecouleur ;
    List<chebyy> tableofdistance ;
    List<hu> tableofhu ;
    List<huu> distancehu ;
    List<llbp> distancelbp ;
    List<distancepobdre> distancefinal ;


    String selectedImagePath;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main7);
        textView = findViewById(R.id.text);
        textView1 = findViewById(R.id.text1);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        Menu navigationView = (Menu) findViewById(R.id.navigation_view);
        mFloatingNavigationView = (FloatingNavigationView) findViewById(R.id.floating_navigation_view);
        imag = findViewById(R.id.imageView6);
        imageVIewOuput = findViewById(R.id.imageView7);
        sqLiteHelper = new SQLiteHelper(this, "ObjectDB.sqlite", null, 1);
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS ORBFEATURE(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS SHAPE(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS CHEBY(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, chebyvect VARCHAR )");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS HU(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, huvect VARCHAR )");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS HUMOMO(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, huvect VARCHAR )");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS CHEBYSEVEN(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, chebyvect VARCHAR )");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS CHEBYNINE(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, chebyvect VARCHAR )");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS COLORF(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");
        //        sqLiteHelper.queryData("select  features from OBJECTFEATURE ");
        sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS LBP(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, t INTEGER, w INTEGER, h INTEGER,image BLOB)");
        // sqLiteHelper.queryData("DELETE FROM LBP WHERE name='banana000' " );
        //  sqLiteHelper.queryData("DELETE FROM  COLORF WHERE name='pear12'" );
        //sqLiteHelper.queryData("DELETE FROM ORBFEATURE");
        //sqLiteHelper.queryData("DELETE FROM SHAPE");
        // sqLiteHelper.queryData("DELETE FROM HUMOMO");
        // sqLiteHelper.queryData("DELETE FROM CHEBYSEVEN WHERE id=471" );
        // sqLiteHelper.queryData("DELETE FROM HU WHERE id=73" );
        //sqLiteHelper.queryData("DELETE FROM CHEBY " );
        imag.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {

                //CropImage.activity().start(MainActivity7.this);
                //selectImage();
                OpenImages();
            }

        });


        mFloatingNavigationView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mFloatingNavigationView.open();
            }
        });
        context = getApplicationContext();
        mFloatingNavigationView.setNavigationItemSelectedListener(new NavigationView.OnNavigationItemSelectedListener() {
            @Override
            public boolean onNavigationItemSelected(MenuItem item) {
                // Snackbar.make((View) mFloatingNavigationView.getParent(), item.getTitle() + " Selected!", Snackbar.LENGTH_SHORT).show();
                switch (item.getItemId()) {
                    // your logic here.
                    case R.id.exctract:
                        deleteCache(context);

                        // Exctraction();
                        AsyncTaskRunner runner = new AsyncTaskRunner(MainActivity7.this);

                        runner.execute();
                        mFloatingNavigationView.getMenu().findItem(R.id.contour).setEnabled(true);
                        mFloatingNavigationView.getMenu().findItem(R.id.reco2).setEnabled(true);
                        mFloatingNavigationView.getMenu().findItem(R.id.lbp).setEnabled(true);

                        clicked = true;
                        return true;
                    case R.id.lbp:


                        if (clicked == true) {

                            //lbpedescriptore();
                            AsyncTaskRunnerelbp runner1 = new AsyncTaskRunnerelbp(MainActivity7.this);

                            runner1.execute();
                        } else {
                            Toast.makeText(getApplicationContext(), "Object extraction  first, please ", Toast.LENGTH_LONG).show();
                            item.setEnabled(false);

                        }
                        return true;



                    case R.id.contour:


                        if (clicked = true) {
                            String s=  Canny();

                            Intent intent = new Intent(MainActivity7.this, Activitycheby.class);
                            intent.putExtra("chebymoment", s);
                            // intent.putExtra("Image", bitmap);
                            //  intent.putExtra("picture", byteArray);
                            intent.putExtra("imgurl", u);
                            startActivity(intent);

                           /* AsyncTaskRunnercheby runnerch = new AsyncTaskRunnercheby(MainActivity7.this);

                            runnerch.execute();*/

                        } else {
                            Toast.makeText(getApplicationContext(), "Object extraction  first, please", Toast.LENGTH_LONG).show();
                            item.setEnabled(false);

                        }
                        return true;

                    case R.id.reco2: {
                        if (clicked == true) {

                            clordescr();

                        } else {
                            Toast.makeText(getApplicationContext(), "Object extraction  first, please", Toast.LENGTH_LONG).show();
                            item.setEnabled(false);

                        } }
                    return true;



                    case R.id.reco:

                        AsyncTaskRunnerepondre runner2 = new AsyncTaskRunnerepondre(MainActivity7.this);

                        runner2.execute();
                        return true;


                }

                mFloatingNavigationView.close();
                return true;
            }
        });
    }

    protected void showInputDialog(final Mat img) {

        // get prompts.xml view
        LayoutInflater layoutInflater = LayoutInflater.from(MainActivity7.this);
        View promptView = layoutInflater.inflate(R.layout.input_dialog, null);
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity7.this);
        alertDialogBuilder.setView(promptView);

        final EditText editText = (EditText) promptView.findViewById(R.id.edittext);
        // setup a dialog window
        alertDialogBuilder.setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //resultText.setText("Hello, " + editText.getText());

                        sqLiteHelper.dbput(
                                editText.getText().toString().trim(),
                                img
                        );
                        Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                dialog.cancel();
                            }
                        });

        // create an alert dialog
        AlertDialog alert = alertDialogBuilder.create();
        alert.show();
    }

    public void alertOneButton(String s) {

        new AlertDialog.Builder(MainActivity7.this)
                .setTitle("Resulat de probabilite")
                .setMessage(s)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {

                        dialog.cancel();
                    }
                }).show();
    }

    @Override
    public void onBackPressed() {
        if (mFloatingNavigationView.isOpened()) {
            mFloatingNavigationView.close();
        } else {
            super.onBackPressed();
        }
    }

    /* @Override
     protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

         super.onActivityResult(requestCode, resultCode, data);


         if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE) {


             CropImage.ActivityResult result = CropImage.getActivityResult(data);

             if (resultCode == RESULT_OK) {

                 mImageUri = result.getUri();

                 imag.setImageURI(mImageUri);
                 try {
                    bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), mImageUri);
                     img_input = new Mat();
                     Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                     Utils.bitmapToMat(bmp32, img_input);

                     textView.setText("Image  traiter ");
                     textView.setTextColor(Color.BLACK);
                 } catch (IOException e) {
                     e.printStackTrace();
                 }

             } else if (resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE) {

                 Exception e = result.getError();

                 Log.d("error", e.toString());

             }




         }

     }*/
    public boolean checkPermission(){

        int result = ContextCompat.checkSelfPermission(MainActivity7.this, Manifest.permission.READ_EXTERNAL_STORAGE);
        int result1 = ContextCompat.checkSelfPermission(MainActivity7.this,Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int result2 = ContextCompat.checkSelfPermission(MainActivity7.this,Manifest.permission.CAMERA);

        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED && result2 == PackageManager.PERMISSION_GRANTED;

    }

    public void requestPermission(){

        ActivityCompat.requestPermissions(MainActivity7.this,new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.CAMERA},PERMISSION_CODE);

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        switch (requestCode){

            case PERMISSION_CODE :

                if (grantResults.length > 0){

                    boolean storage  = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean cameras = grantResults[0] == PackageManager.PERMISSION_GRANTED;

                    if (storage && cameras){

                        Toast.makeText(MainActivity7.this, "Permission Granted", Toast.LENGTH_SHORT).show();

                    }else{

                        Toast.makeText(MainActivity7.this, "Permission Denied", Toast.LENGTH_SHORT).show();

                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){

                            if (shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE)){

                                showMsg("You need to allow access to the permissions", new DialogInterface.OnClickListener() {
                                    @Override
                                    public void onClick(DialogInterface dialog, int which) {

                                        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){
                                            requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.CAMERA},PERMISSION_CODE);
                                        }
                                    }
                                });
                                return;
                            }
                        }
                    }
                }
        }

    }

    private void showMsg(String s, DialogInterface.OnClickListener listener) {

        new AlertDialog.Builder(MainActivity7.this)
                .setMessage(s)
                .setPositiveButton("OK", listener)
                .setNegativeButton("Cancel", null)
                .create()
                .show();
    }

    private void OpenImages() {

        final CharSequence[] option = {"Camera","Gellary"};

        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity7.this);
        builder.setTitle("Select Action");
        builder.setItems(option, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {

                if (option[which].equals("Camera")){
                    CameraIntent();
                }
                if (option[which].equals("Gellary")){
                    GellaryIntent();

                }
            }
        });

        AlertDialog dialog = builder.create();
        dialog.show();

    }

    private void GellaryIntent() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select File"),gellary);
    }

    private void CameraIntent() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent,camera);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK){

            if (requestCode == camera){
                try { OpenCameraResult(data);
                } catch (Exception e) {
                    e.printStackTrace();
                }
            }else if (requestCode == gellary){
                OpenGellaryResult(data);
            }
        }
    }

    private void OpenGellaryResult(Intent data) {

        bitmap = null;

        if (data != null){
            try {u = data.getData();
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(),data.getData());
                ByteArrayOutputStream stream = new ByteArrayOutputStream();
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                byteArray = stream.toByteArray();
                int orientation = getOrientationOfImage(path); // 런타임 퍼미션 필요

                bitmap = getRotatedBitmap( bitmap, orientation);
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        imag.setImageBitmap(bitmap);
        img_input = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, img_input);
        textView.setText("Source image ");
        textView.setTextColor(Color.BLACK);


    }
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


    private void OpenCameraResult(Intent data) throws Exception {
        //u = data.getData();
        bitmap = (Bitmap) data.getExtras().get("data");
        rotateimage(bitmap);

        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, bytes);
        bitmap = BitmapFactory.decodeStream(new ByteArrayInputStream(bytes.toByteArray()));

        File paths = new File(Environment.getExternalStorageDirectory(), System.currentTimeMillis() + ".jpg");

        Toast.makeText(this, "Path -> " + paths.getAbsolutePath(), Toast.LENGTH_SHORT).show();
        Log.d("tag","File Path -> " + paths.getName());

        try {
            FileOutputStream fos = new FileOutputStream(paths);
            paths.createNewFile();

            if(!paths.exists()){
                paths.mkdir();
            }

            fos.write(bytes.toByteArray());
            fos.close();
            String realPath = paths.getAbsolutePath();
            File f = new File(realPath);
            u = Uri.fromFile(f);
        } catch (IOException e) {
            e.printStackTrace();
        }

        textView.setText("Source image ");
        textView.setTextColor(Color.BLACK);
        imag.setImageBitmap(bitmap);
        img_input = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, img_input);

    }
    private void rotateimage(Bitmap bitmap) throws IOException {
        ExifInterface exifInterface=null;
        exifInterface =new ExifInterface(path);
        int orientation =exifInterface.getAttributeInt(ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_UNDEFINED);
        Matrix matrix =new Matrix();
        switch(orientation){

            case ExifInterface.ORIENTATION_ROTATE_90:
                matrix.setRotate(90);
                break ;
            case ExifInterface.ORIENTATION_ROTATE_180:
                matrix.setRotate(180);
                break ;
            default:
        }
        Bitmap rptatedimage =Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);

    }
    public native void nGRIS(long inputImage1, long outputImage1);

    private void NiveauGris() {
        Mat gris;
        gris = new Mat();

        Bitmap bitmapOutput = null;
        // excract(img_input.getNativeObjAddr(), out.getNativeObjAddr());


        nGRIS(img_output.getNativeObjAddr(), gris.getNativeObjAddr());
        bitmapOutput = Bitmap.createBitmap(gris.cols(), gris.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(gris, bitmapOutput);
        imageVIewOuput.setImageBitmap(bitmapOutput);

        textView1.setText("image en niveau de gris ");
        textView1.setTextColor(Color.BLACK);
        gris = null;
        imageVIewOuput.invalidate();
    }

    public native void paraellegrabcut(long inputImage, long outputImage);

    public native void filtregaussin(long inputImage2, long outputImage2);

    private void filter() {
        Mat filter1;
        filter1 = new Mat();
        Bitmap bitmapOutput1 = null;
        // excract(img_input.getNativeObjAddr(),  filtinput.getNativeObjAddr());


        // filtregaussin(img_output.getNativeObjAddr(),filter1.getNativeObjAddr());
        paraellegrabcut(img_output.getNativeObjAddr(), filter1.getNativeObjAddr());
        bitmapOutput1 = Bitmap.createBitmap(filter1.cols(), filter1.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(filter1, bitmapOutput1);
        imageVIewOuput.setImageBitmap(bitmapOutput1);

        textView1.setText("image apres filtre gaussien");
        textView1.setTextColor(Color.BLACK);

        filter1 = null;
        imageVIewOuput.invalidate();
    }

    public static double calculateDistance(double[] array1, double[] array2) {
        double Sum = 0.0;
        for (int i = 0; i < array1.length; i++) {
            Sum = Sum + Math.pow((array1[i] - array2[i]), 2.0);
        }
        return Math.sqrt(Sum);
    }

    public native double[] humom(long inputImage, long outputImage);

    public native void remover(long inputImage, long outputImage);

    public native void watershed(long inputImage, long outputImage);

    public native void grabcut(long inputImage, long outputImage);

    public native double shapedescr(long inputImag, long outputImag);

    public native void excract(long inputImage, long outputImage);

    private void Exctraction() {

        //   img_output = new Mat();
        Mat bb1 = new Mat();
        Mat out = new Mat();
        Bitmap bitmapOutput2 = null;
        //excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        String HUvct="";
        double [] h1= humom(img_output.getNativeObjAddr(),out.getNativeObjAddr());

        for (int i = 0; i < h1.length; i++) {
            HUvct+=h1[i]+" ";
            // Toast.makeText(getApplicationContext(), "distance !" +  HUvct, Toast.LENGTH_LONG).show();
        }
        bitmapOutput2 = Bitmap.createBitmap(img_output.cols(), img_output.rows(), Bitmap.Config.ARGB_8888);

        // final File file = new File(path, "example.txt");

        // Save your stream, don't forget to flush() it before closing it.

        /*
         * */
        Utils.matToBitmap(img_output, bitmapOutput2);
        imageVIewOuput.setImageBitmap(bitmapOutput2);
        textView1.setText("image after extraction ");
        textView1.setTextColor(Color.BLACK);
        // showInputDialog(img_output);
        InputDialogShaper(HUvct,img_output);
    /* bb1= sqLiteHelper.dbgetshape("tomate1");
       double dis= shapedescr(img_output.getNativeObjAddr(),bb1.getNativeObjAddr());

         Toast.makeText(getApplicationContext(), "distance "+dis, Toast.LENGTH_SHORT).show();*/

        imageVIewOuput.invalidate();
    }

    public native void edge(long inputImag, long outputImag);

    public native void DetectEdge(long inputImage3, long outputImage3);

    public native float[] bychev(long inputImage3);


    public static float eculiddistance(float a[], float b[]) {
        float distance;
        float somme = 0.0f;
        for (int i = 0; i < a.length; i++) {
            somme +=  Math.pow((a[i] - b[i]), 2);
        }
        distance= (float) Math.sqrt(somme);

        return distance;
    }
    private String Canny(){

        Mat can;
        Mat caninput =new Mat();
        can = new Mat();
        Bitmap bitmapOutput3=null;
        String cheby="";
        excract(img_input.getNativeObjAddr(),  caninput.getNativeObjAddr());
        String chebyvct="";
        float []  b1= bychev(caninput.getNativeObjAddr());
        DetectEdge(caninput.getNativeObjAddr(), can.getNativeObjAddr());
        for (int i = 0; i < b1.length; i++) {
            chebyvct+=b1[i]+" ";
            cheby+=b1[i]+" , ";

            // Toast.makeText(getApplicationContext(), "distance !" +  chebyvct, Toast.LENGTH_LONG).show();
        }


     /*   Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM CHEBY  ");

        if (cursor.moveToFirst()) {
            while (!cursor.isAfterLast()) {
                String s = cursor.getString(2);

                String fdata[] = s.split(" ");
                float  array_f[]= new float[fdata.length];
                for (int i = 0; i < fdata.length; i++) {
                    {
                        array_f[i] = Float.parseFloat(fdata[i]);
                    }
                    float d=eculiddistance(b1,array_f);
                    Toast.makeText(getApplicationContext(), "distance !" + d, Toast.LENGTH_LONG).show();
                    cursor.moveToNext();
                }
            }
        }

        bitmapOutput3  = Bitmap.createBitmap(caninput.cols(), caninput.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(caninput, bitmapOutput3);
        imageVIewOuput.setImageBitmap(bitmapOutput3);
        textView1.setText("Cheybyshev moments ");
        textView1.setTextColor(Color.BLACK);
       // chebudialog(chebyvct);
        can=null;*/
        return cheby;
    }
    protected void chebudialog(final String CHB) {

        // get prompts.xml view
        LayoutInflater layoutInflater = LayoutInflater.from(MainActivity7.this);
        View promptView = layoutInflater.inflate(R.layout.input_dialog, null);
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity7.this);
        alertDialogBuilder.setView(promptView);

        final EditText editText = (EditText) promptView.findViewById(R.id.edittext);
        // setup a dialog window
        alertDialogBuilder.setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //resultText.setText("Hello, " + editText.getText());

                        sqLiteHelper.insertcheby(
                                editText.getText().toString().trim(),
                                CHB
                        );
                        Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                dialog.cancel();
                            }
                        });

        // create an alert dialog
        AlertDialog alert = alertDialogBuilder.create();
        alert.show();
    }
    public native void orbcall(long inputImage4, long outputImage4);
    public native void orb(long inputImage4, long outputImage4,long descriptorr);
    private void orbdescriptor(){

        Mat deorb;
        deorb = new Mat();
        Bitmap bitmapOutput4 =null;
        Mat descr=new Mat();
        orb(img_output.getNativeObjAddr(), deorb.getNativeObjAddr(),descr.getNativeObjAddr());
        bitmapOutput4 = Bitmap.createBitmap(deorb.cols(),deorb.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(deorb, bitmapOutput4);
        imageVIewOuput.setImageBitmap(bitmapOutput4);
        textView1.setText("descriptors orb ");
        textView1.setTextColor(Color.BLACK);
       /* Cursor cursor1 = this.sqLiteHelper.getData("SELECT * FROM ORBFEATURE  ");
        int count ;
        if (cursor1.moveToFirst()) {
            while (!cursor1.isAfterLast()) {
                int t = cursor1.getInt(2);
                int w = cursor1.getInt(3);
                int h = cursor1.getInt(4);
                byte[] p = cursor1.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);


                if(bb.size()!=descr.size()) {
                    Imgproc.resize(bb, bb, descr.size(), 0.5, 0.5, Imgproc.INTER_AREA);
                }
                Mat result = new Mat();
                // Core.bitwise_xor(bb , img_output  , result);
                //   int similarPixels  = countNonZero(result);
                double dist_ham = Core.norm(descr,bb,NORM_HAMMING2);
                String ss=Double.toString(dist_ham );
                alertOneButton(ss);
                //  double dist_ham = Core.norm(bb1,bb,NORM_HAMMING2);
                // String ss=Double.toString(dist_ham);
                count=0;
                /*
                for(int i =0; i<h;i++){
                    for(int j=0;j<w ;j++){

                        if(!bb.equals(img_output)){
                            count++;
                        }
                    }
                }
*/

            /*    cursor1.moveToNext();
            }
        }*/
        //  showInputDialog(descr);
        deorb=null;
        imageVIewOuput.invalidate();

    }

    public native double chisquare(long inputImage, long outputImage) ;
    public native void hsvhisto(long inputImage, long outputImage);
    public native int[] lbp(long inputImage5, long outputImage5);
    public native String lbpshiiit(long inputImage5, long outputImage5);
    public native void lbpfunc(long inputImage, long outputImage);
    public native void elbbp(long inputImage, long outputImage,long hist);

    private void lbpedescriptore(){
        Mat delbp;
        delbp = new Mat();
        Mat hist=new Mat();
        lbpinput = new Mat();
        excract(img_input.getNativeObjAddr(),  lbpinput.getNativeObjAddr());
        Bitmap bitmapOutput5 =null;

//String msg=  lbpshiiit(lbpinput.getNativeObjAddr(),  delbp.getNativeObjAddr());
        elbbp(lbpinput.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
        bitmapOutput5 = Bitmap.createBitmap( delbp.cols(), delbp.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap( delbp, bitmapOutput5);
        imageVIewOuput.setImageBitmap(bitmapOutput5);
        textView1.setText("Elbp result ");
        textView1.setTextColor(Color.BLACK);
    /*    final File file = new File(path, "example.txt");

        // Save your stream, don't forget to flush() it before closing it.

        try
        {
            file.createNewFile();
            FileOutputStream fOut = new FileOutputStream(file);
           OutputStreamWriter myOutWriter = new OutputStreamWriter(fOut);
          /*   myOutWriter.append(msg);

            myOutWriter.close();
            for (int i = 0; i < b1.length; i++) {
                myOutWriter.write(b1[i] + "\t"+ "");
            }myOutWriter.close();
            fOut.flush();
            fOut.close();
        }
        catch (IOException e)
        {
            Log.e("Exception", "File write failed: " + e.toString());
        }

        // Toast.makeText(getApplicationContext(), "distance "+s, Toast.LENGTH_SHORT).show();
        hist.convertTo(hist, CV_8U);
     Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM LBP  ");

      if (cursor.moveToFirst()) {
            while (!cursor.isAfterLast()) {
                String name =cursor.getString(1);
                int t = cursor.getInt(2);
                int w = cursor.getInt(3);
                int h = cursor.getInt(4);
                byte[] p = cursor.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);
               bb.convertTo(bb, CV_32F);
                hist.convertTo(hist, CV_32F);

               double res = Imgproc.compareHist(hist,bb, Imgproc.CV_COMP_CHISQR);
               Double d = new Double(res * 100);
                String ss=Double.toString(d);
                //alertOneButton(ss);
                Toast.makeText(getApplicationContext(), "namee !"+d, Toast.LENGTH_LONG).show();
                cursor.moveToNext();
            }
        }
*/
        hist.convertTo(hist, CV_8U);
        // showInputDialog(hist);

        delbp=null;
        imageVIewOuput.invalidate();
    }
    public final void saveMat(String path, Mat mat) {
        File file = new File(path).getAbsoluteFile();
        file.getParentFile().mkdirs();
        try {
            int cols = mat.cols();
            float[] data = new float[(int) mat.total() * mat.channels()];
            mat.get(0, 0, data);
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(path))) {
                oos.writeObject(cols);
                oos.writeObject(data);
                oos.close();
            }
        } catch (IOException | ClassCastException ex) {
            System.err.println("ERROR: Could not save mat to file: " + path);
            //Logger.getLogger(this.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public native void histths(long inputImage, long outputImage);
    public native void calculcolordescriptors(long inputImage, long outputImage);
    public native void dominatwithkmeans(long inputImage, long outputImage);
    public native int hsvhistt(long inputImage);
    public native void fuckhsv(long inputImage, long outputImage);
    public void clordescr(){
        Mat decol;
        decol = new Mat();Mat can1;
        can1=new Mat();
        // Mat img_output ;
        // img_output=new Mat();

        //  excract(img_input.getNativeObjAddr(),  img_output.getNativeObjAddr());

        // DetectEdge(img_output.getNativeObjAddr(), can1.getNativeObjAddr());
        Bitmap bitmapOutput5 =null;
        histths(img_output.getNativeObjAddr(),decol.getNativeObjAddr());
        fuckhsv(img_output.getNativeObjAddr(),can1.getNativeObjAddr());
        // dominatwithkmeans(img_output.getNativeObjAddr(),  decol.getNativeObjAddr());
        //  int a=hsvhistt(img_output.getNativeObjAddr());
        // Toast.makeText(getApplicationContext(), "value !"+a, Toast.LENGTH_LONG).show();
       /* Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM COLORF  ");

        if (cursor.moveToFirst()) {
            while (!cursor.isAfterLast()) {
                String name =cursor.getString(1);
                int t = cursor.getInt(2);
                int w = cursor.getInt(3);
                int h = cursor.getInt(4);
                byte[] p = cursor.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);
                Mat b1=new Mat();

                histths(bb.getNativeObjAddr(),b1.getNativeObjAddr());
                b1.convertTo(b1,CV_32F);
                decol.convertTo(decol,CV_32F);
                double d = Imgproc.compareHist(decol,b1, Imgproc.CV_COMP_BHATTACHARYYA);

                // double d=shapedescr(img_output.getNativeObjAddr(),bb.getNativeObjAddr());
                //alertOneButton(ss);
                Toast.makeText(getApplicationContext(), name+" " +d, Toast.LENGTH_LONG).show();
                cursor.moveToNext();
            }
        }*/

        bitmapOutput5 = Bitmap.createBitmap( can1.cols(), can1.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap( can1, bitmapOutput5);

        imageVIewOuput.setImageBitmap(bitmapOutput5);
        textView1.setText("Hue histograme ");
        textView1.setTextColor(Color.BLACK);
        decol.convertTo(decol, CV_8U);
        //   InputDialogColor(decol,can1);
        decol=null;
        imageVIewOuput.invalidate();



    }
    private class AsyncTaskRunner extends AsyncTask<String, String, Bitmap> {

        private String resp;
        private ProgressDialog dialog;
        public AsyncTaskRunner(MainActivity7 activity) {
            dialog = new ProgressDialog(activity);
        }
        @Override
        protected void onPreExecute() {
            dialog.setMessage("Extraction on progress, please wait.");
            dialog.setCancelable(false);
            dialog.show();
        }
        @Override
        protected Bitmap doInBackground(String...string) {
            // Some long-running task like downloading an image.
            img_output = new Mat();
            Mat bb1 = new  Mat();
            Mat out=new Mat();
            Bitmap bitmapOutput2 =null;
            excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());



            bitmapOutput2 = Bitmap.createBitmap(img_output.cols(), img_output.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(img_output, bitmapOutput2);


            return bitmapOutput2;
        }


        @Override
        protected void onPostExecute(Bitmap result) {
            // This method is executed in the UIThread
            // with access to the result of the long running task
            if (dialog.isShowing()) {
                dialog.dismiss();
            }
            imageVIewOuput.setImageBitmap(result);
            textView1.setText("image after extraction ");

        }
    }
    public static void deleteCache(Context context) {
        try {
            File dir = context.getCacheDir();
            deleteDir(dir);
        } catch (Exception e) { e.printStackTrace();}
    }

    public static boolean deleteDir(File dir) {
        if (dir != null && dir.isDirectory()) {
            String[] children = dir.list();
            for (int i = 0; i < children.length; i++) {
                boolean success = deleteDir(new File(dir, children[i]));
                if (!success) {
                    return false;
                }
            }
            return dir.delete();
        } else if(dir!= null && dir.isFile()) {
            return dir.delete();
        } else {
            return false;
        }
    }
    private class AsyncTaskRunnerelbp extends AsyncTask<String, String, Mat> {

        private String resp;
        private ProgressDialog dialog;
        public AsyncTaskRunnerelbp(MainActivity7 activity) {
            dialog = new ProgressDialog(activity);
        }
        @Override
        protected void onPreExecute() {
            dialog.setMessage("Doing something, please wait.");
            dialog.setCancelable(false);
            dialog.show();
        }
        @Override
        protected Mat doInBackground(String...string) {
            // Some long-running task like downloading an image.
            Mat delbp;
            delbp = new Mat();
            Mat hist=new Mat();
            lbpinput = new Mat();
            ///excract(img_input.getNativeObjAddr(),  lbpinput.getNativeObjAddr());
            Bitmap bitmapOutput5 =null;

//String msg=  lbpshiiit(lbpinput.getNativeObjAddr(),  delbp.getNativeObjAddr());
            elbbp(img_output.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
            bitmapOutput5 = Bitmap.createBitmap( delbp.cols(), delbp.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap( delbp, bitmapOutput5);
            hist.convertTo(hist, CV_8U);


            return hist;
        }


        @Override
        protected void onPostExecute(Mat result) {
            // This method is executed in the UIThread
            // with access to the result of the long running task
            if (dialog.isShowing()) {
                dialog.dismiss();
            }
            showInputDialog(result);

        }
    }


    /*
    private void selectImage() {
        final CharSequence[] options = { "Take Photo", "Choose from Gallery","Cancel" };
        AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity7.this);
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
                   imag.setImageBitmap(bitmap);
                    img_input = new Mat();
                    Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Utils.bitmapToMat(bmp32, img_input);
                    Size sz = new Size(100,100);
                    Imgproc.resize( img_input, img_input, sz );
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
               imag.setImageBitmap(bitmap);

                img_input = new Mat();
                Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                Utils.bitmapToMat(bmp32, img_input);

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


*/
    List<llbp>   RecognitionELbp(){


        Mat delbp;
        delbp = new Mat();
        Mat hist=new Mat();
        lbpinput = new Mat();
        excract(img_input.getNativeObjAddr(),  lbpinput.getNativeObjAddr());
        Bitmap bitmapOutput5 =null;
        elbbp(lbpinput.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
        distancelbp = new ArrayList<llbp>();
        Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM LBP  ");

        if (cursor.moveToFirst()) {
            while (!cursor.isAfterLast()) {
                String name =cursor.getString(1);
                int t = cursor.getInt(2);
                int w = cursor.getInt(3);
                int h = cursor.getInt(4);
                byte[] p = cursor.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);
                bb.convertTo(bb, CV_32F);
                hist.convertTo(hist, CV_32F);

                double res = w2*Imgproc.compareHist(hist,bb, Imgproc.CV_COMP_CHISQR);
                Double d = new Double(res * 100);
                String s1 = cursor.getString(1);
                llbp foo = new llbp(d, s1);
                distancelbp.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();
        // Collections.sort(distancelbp, new Sortbyroll1());




        /*for (llbp a: distancelbp) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }*/
        return distancelbp;
    }
    List<couleur>   RecognitionColor(){


        Mat decol;
        decol = new Mat();Mat can1;
        can1=new Mat();distancecouleur = new ArrayList<couleur>();
      /*  Mat img_output ;
        img_output=new Mat();

Mat img=new Mat();
        excract(img_input.getNativeObjAddr(),  img_output.getNativeObjAddr());

        DetectEdge(img_output.getNativeObjAddr(), can1.getNativeObjAddr());*/
        histths(img_output.getNativeObjAddr(),decol.getNativeObjAddr());

        Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM COLORF  ");

        if (cursor.moveToFirst()) {
            while (!cursor.isAfterLast()) {
                String name =cursor.getString(1);
                int t = cursor.getInt(2);
                int w = cursor.getInt(3);
                int h = cursor.getInt(4);
                byte[] p = cursor.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);

                bb.convertTo(bb,CV_32F);
                decol.convertTo(decol,CV_32F);
                double d = w3*Imgproc.compareHist(decol,bb, Imgproc.CV_COMP_BHATTACHARYYA);

                String s1 = cursor.getString(1);
                couleur foo = new couleur(d, s1);
                distancecouleur.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();
       /* Cursor cursor1 = this.sqLiteHelper.getData("SELECT * FROM SHAPE  ");

        if (cursor1.moveToFirst()) {
            while (!cursor1.isAfterLast()) {
                int t = cursor1.getInt(2);
                int w = cursor1.getInt(3);
                int h = cursor1.getInt(4);
                byte[] p = cursor1.getBlob(5);
                Mat bb = new Mat(h,w,t);
                bb.put(0,0,p);
Mat b1=new Mat();

                // double d1=Imgproc.matchShapes(bb,img_output,Imgproc.CONTOURS_MATCH_I1,0);

                double []  h2=  humom(bb.getNativeObjAddr(),b1.getNativeObjAddr());

                // double d=Imgproc.matchShapes(b1,out,Imgproc.CONTOURS_MATCH_I1,0);
                double d=calculateDistance(h1,h2);
                String s1 = cursor1.getString(1);
               shapeu foo1 = new shapeu(d, s1);
                distanceshape.add(foo1);
                cursor1.moveToNext();
            }
            cursor1.close();
        }*/
        //Collections.sort(distancecouleur, new Sortbyroll2());
        // Collections.sort(distanceshape, new Sortbyroll3());
        //for (int i=0; i<38; i++)



       /* for (couleur a: distancecouleur) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }
        for (shapeu b: distanceshape) {

            Log.v("Array Value","Array Value"+b.namobject+" "+b.distance);
        }*/
        return distancecouleur;
    }
    List<chebyy>  Recognitioncheby(){


        Mat caninput =new Mat();

        Bitmap bitmapOutput3=null;
        tableofdistance = new ArrayList<chebyy>();
        excract(img_input.getNativeObjAddr(),  caninput.getNativeObjAddr());
        String chebyvct="";
        float []  b1= bychev(caninput.getNativeObjAddr());

        for (int i = 0; i < b1.length; i++) {
            chebyvct+=b1[i]+" ";
            // Toast.makeText(getApplicationContext(), "distance !" +  chebyvct, Toast.LENGTH_LONG).show();
        }
        //  chebudialog(chebyvct);

        Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM CHEBYSEVEN  ");

        for(cursor.moveToFirst();!cursor.isAfterLast();cursor.moveToNext())
        {  String s = cursor.getString(2);

            String fdata[] = s.split(" ");
            float  array_f[]= new float[fdata.length];
            for (int i = 0; i < fdata.length; i++) {
                {
                    array_f[i] = Float.parseFloat(fdata[i]);
                }

                double d= (double) (w1*eculiddistance(b1,array_f));
                String s1 = cursor.getString(1);
                chebyy foo = new chebyy(d, s1);
                tableofdistance.add(foo);
                //  cursor.moveToNext();


            }}
        cursor.close();
        Collections.sort(tableofdistance, new Sortbyroll());
        //for (int i=0; i<38; i++)



      /*for (chebyy a: tableofdistance) {

          Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
      }*/
        return tableofdistance;
    }
    List<hu>  Recognitionhu(){


        img_output = new Mat();
        Mat bb1 = new Mat();
        Mat out = new Mat();
        tableofhu = new ArrayList<hu>();
        Bitmap bitmapOutput2 = null;
        excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        String HUvct="";
        double [] h1= humom(img_output.getNativeObjAddr(),out.getNativeObjAddr());


        Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM HU  ");

        for(cursor.moveToFirst();!cursor.isAfterLast();cursor.moveToNext())
        {  String s = cursor.getString(2);

            String fdata[] = s.split(" ");
            double  array_f[]= new double[fdata.length];
            for (int i = 0; i < fdata.length; i++) {
                {
                    array_f[i] = Double.parseDouble(fdata[i]);
                }

                double d= calculateDistance(h1,array_f);
                String s1 = cursor.getString(1);
                hu foo2 = new hu(d, s1);
                tableofhu.add(foo2);
                // cursor.moveToNext();


            }}
        cursor.close();

     /*   Collections.sort(tableofhu, new Sortbyrollhu());
        //for (int i=0; i<38; i++)



      for (hu a: tableofhu) {

          Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
      }*/
        return tableofhu;
    }

    protected void InputDialogColor(final Mat img,final Mat img1) {

        // get prompts.xml view
        LayoutInflater layoutInflater = LayoutInflater.from(MainActivity7.this);
        View promptView = layoutInflater.inflate(R.layout.input_dialog, null);
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity7.this);
        alertDialogBuilder.setView(promptView);

        final EditText editText = (EditText) promptView.findViewById(R.id.edittext);
        // setup a dialog window
        alertDialogBuilder.setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //resultText.setText("Hello, " + editText.getText());

                      /*  sqLiteHelper.dbputcolor(
                                editText.getText().toString().trim(),
                                img
                        );*/
                        sqLiteHelper.dbputshape(
                                editText.getText().toString().trim(),
                                img1
                        );
                        Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                dialog.cancel();
                            }
                        });

        // create an alert dialog
        AlertDialog alert = alertDialogBuilder.create();
        alert.show();
    }
    List<huu>  Recognitionhuu(){


        // img_output = new Mat();
        Mat bb1 = new Mat();
        Mat out = new Mat();
        distancehu = new ArrayList<huu>();
        Bitmap bitmapOutput2 = null;
        // excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        String HUvct="";
        double [] h1= humom(img_output.getNativeObjAddr(),out.getNativeObjAddr());


        Cursor cursor = this.sqLiteHelper.getData("SELECT * FROM HUMOMO  ");

        for(cursor.moveToFirst();!cursor.isAfterLast();cursor.moveToNext())
        {  String s = cursor.getString(2);

            String fdata[] = s.split(" ");
            double  array_f[]= new double[fdata.length];
            for (int i = 0; i < fdata.length; i++)
            {
                array_f[i] = Double.parseDouble(fdata[i]);
            }

            double d= calculateDistance(h1,array_f);
            String s1 = cursor.getString(1);
            huu foo3= new huu(d, s1);
            distancehu.add(foo3);
            // cursor.moveToNext();


        }
        cursor.close();

        //Collections.sort(distancehu, new Sortbyrollhuu());
        //for (int i=0; i<38; i++)



       /*for (huu a: distancehu) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }*/
        return distancehu;
    }
    protected void InputDialogShaper(final String HUU, final Mat img1) {

        // get prompts.xml view
        LayoutInflater layoutInflater = LayoutInflater.from(MainActivity7.this);
        View promptView = layoutInflater.inflate(R.layout.input_dialog, null);
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity7.this);
        alertDialogBuilder.setView(promptView);

        final EditText editText = (EditText) promptView.findViewById(R.id.edittext);
        // setup a dialog window
        alertDialogBuilder.setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //resultText.setText("Hello, " + editText.getText());

                        sqLiteHelper.inserthu1(
                                editText.getText().toString().trim(),
                                HUU

                        );
                        /*sqLiteHelper.dbputshape(
                                editText.getText().toString().trim(),
                                img1
                        );*/
                        Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                dialog.cancel();
                            }
                        });

        // create an alert dialog
        AlertDialog alert = alertDialogBuilder.create();
        alert.show();
    }


    class Thread0 extends Thread {

        @Override
        public void run() {

            distancecouleur=RecognitionColor();

        }


    }
    class Thread1 extends Thread {

        @Override
        public void run() {

            distancelbp=RecognitionELbp();

        }
    }
    class Thread2 extends Thread {

        @Override
        public void run() {

            distancehu= Recognitionhuu();

        }


    }

    String distanceponde(List<couleur> a,List<huu> b,List<llbp> c) {
        double distance;
        String nom;
        String dis="";
        distancefinal = new ArrayList<distancepobdre>();
        for (int i = 0; i < a.size();i++) {
            nom=a.get(i).namobject;
            distance = a.get(i).distance+c.get(i).distance+b.get(i).distance;
            distancepobdre ff = new distancepobdre(distance, nom);
            distancefinal.add(ff);
        }
        Collections.sort(distancefinal, new Sortbyroll4());
        Log.v("shit","***********************************************************************");
        for (distancepobdre d: distancefinal) {
            dis+=d.namobject+" , ";
            // Log.v("Array Value","Array Value"+d.namobject+" "+d.distance);
        }
        String fdata[] = dis.split(" , ");
        String res="";
        res+=fdata[0]+" , "+fdata[1]+" , "+fdata[2]+" , "+fdata[3] ;
        return  res;
    }


    /****************************************************************/
    private class AsyncTaskRunnerepondre extends AsyncTask<String, String, String> {

        private String resp;
        private ProgressDialog dialog;
        public AsyncTaskRunnerepondre(MainActivity7 activity) {
            dialog = new ProgressDialog(activity);
        }
        @Override
        protected void onPreExecute() {
            dialog.setMessage("Recognition on progress, please wait.");
            dialog.setCancelable(false);
            dialog.show();
        }
        @Override
        protected String doInBackground(String...string) {
            // Some long-running task like downloading an image.
            t= new Thread0();
            t1 = new Thread1();
            t2 = new Thread2();
            t.start();
            t1.start();
            t2.start();
            try {

                t.join();


            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            try {

                t1.join();

            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            try {

                t2.join();

            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            String s1=distanceponde(distancecouleur,distancehu,distancelbp);
            return  s1;
        }


        @Override
        protected void onPostExecute(String result) {
            // This method is executed in the UIThread
            // with access to the result of the long running task
            if (dialog.isShowing()) {
                dialog.dismiss();
            }

            Intent intent = new Intent(MainActivity7.this, ReActivity.class);
            intent.putExtra("recognition", result);
            // intent.putExtra("Image", bitmap);
            intent.putExtra("imgurl", u);
            startActivity(intent);

        }
    }





    /*************************************************************************/
    void insertgeneral(){

        Mat out = new Mat();
        String HUvct="";
        double [] h1= humom(img_output.getNativeObjAddr(),out.getNativeObjAddr());

        for (int i = 0; i < h1.length; i++) {
            HUvct+=h1[i]+" ";

        }
        String chebyvct="";
        float []  b1= bychev(img_output.getNativeObjAddr());

        for (int i = 0; i < b1.length; i++) {
            chebyvct+=b1[i]+" ";
            //   Toast.makeText(getApplicationContext(), "distance !" +  chebyvct, Toast.LENGTH_LONG).show();
        }
        Mat decol;
        decol = new Mat();
        histths(img_output.getNativeObjAddr(),decol.getNativeObjAddr());
        Mat delbp;
        delbp = new Mat();
        Mat hist=new Mat();
        elbbp(img_output.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
        decol.convertTo(decol, CV_8U);
        hist.convertTo(hist, CV_8U);
        insertthree(HUvct,decol,hist, chebyvct);
    }

    private void insertthree(final String hUvct, final Mat decol, final Mat hist,final String CHB) {

        LayoutInflater layoutInflater = LayoutInflater.from(MainActivity7.this);
        View promptView = layoutInflater.inflate(R.layout.input_dialog, null);
        AlertDialog.Builder alertDialogBuilder = new AlertDialog.Builder(MainActivity7.this);
        alertDialogBuilder.setView(promptView);

        final EditText editText = (EditText) promptView.findViewById(R.id.edittext);
        // setup a dialog window
        alertDialogBuilder.setCancelable(false)
                .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int id) {
                        //resultText.setText("Hello, " + editText.getText());

                        sqLiteHelper.inserthu1(
                                editText.getText().toString().trim(),
                                hUvct

                        );
                        sqLiteHelper.dbputcolor(
                                editText.getText().toString().trim(),
                                decol
                        );
                        sqLiteHelper.dbput(
                                editText.getText().toString().trim(),
                                hist
                        );
                        sqLiteHelper.insertchebyseven(
                                editText.getText().toString().trim(),
                                CHB
                        );
                        Toast.makeText(getApplicationContext(), "ADD!", Toast.LENGTH_SHORT).show();
                    }
                })
                .setNegativeButton("Cancel",
                        new DialogInterface.OnClickListener() {
                            public void onClick(DialogInterface dialog, int id) {
                                dialog.cancel();
                            }
                        });

        // create an alert dialog
        AlertDialog alert = alertDialogBuilder.create();
        alert.show();
    }
}

