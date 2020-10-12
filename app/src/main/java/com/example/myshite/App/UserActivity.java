package com.example.myshite.App;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import  com.example.myshite.MainActivity7;
import android.Manifest;
import android.app.AlertDialog;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
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
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;



import com.example.myshite.R;

import com.example.myshite.ReActivity;
import com.google.android.material.bottomsheet.BottomSheetDialog;

import org.opencv.android.Utils;
import org.opencv.core.Mat;

import org.opencv.imgproc.Imgproc;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;


import static com.example.myshite.MainActivity7.calculateDistance;
import static org.opencv.core.CvType.CV_32F;

public class UserActivity extends AppCompatActivity {

    double w3=0.6355;//couleur
    double w1=0.3052;//chebychev
    double w2=0.5842;//elbp
    Thread t=null;
    private int camera = 1,gellary = 2;

    public static final int PERMISSION_CODE = 111;
    Thread t1=null;
    Thread t2=null;
    TextView textView, textView1;
    private TextView textViewResult;
    private static final int INPUT_SIZE = 224;
    private ImageView imageViewResult;
    private Button bt1;
    private Mat img_output, out, filtinput, canyinput, orbinput, lbpinput;
    private Mat img_input, matdegris, matdecanny, matdeorb, matdelbp, matfilter;
    Uri mImageUri;
    Bitmap bitmap;
    byte[] byteArray;
    Uri u ;
    List<couleur1> distancecouleur ;

Button buttonShow;
    List<huu1> distancehu ;
    List<llbp1> distancelbp ;
    List<distancepobdre1> distancefinal ;
    String path = "/mnt/sdcard/lbp/example.txt";
    String selectedImagePath;
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_user);
       // textViewResult = findViewById(R.id.textView2);
        textView = findViewById(R.id.text);
        textView1 = findViewById(R.id.text1);
        imageViewResult = findViewById(R.id.imageView8);
        buttonShow=findViewById(R.id.buttons);
        buttonShow.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
// here you can bind bottom sheet dialog and initialize Theme in bottom sheet dialog
                final BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(UserActivity.this,
                        R.style.BottomSheetDialogTheme);
// here you can inflate layout which will be shows in bottom sheet dialog
                View bottomSheetView = LayoutInflater.from(getApplicationContext())
                        .inflate(R.layout.bottom_sheet_dialog,
                                (LinearLayout)findViewById(R.id.bottom_sheet_container));
                bottomSheetView.findViewById(R.id.button_share).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Toast.makeText(UserActivity.this, "Share....", Toast.LENGTH_SHORT).show();
                        bottomSheetDialog.dismiss();
                    }
                });
                bottomSheetDialog.setContentView(bottomSheetView);
                bottomSheetDialog.show();

            }
        });
/*bt1.setOnClickListener(new View.OnClickListener() {

    @Override

    public void onClick(View v) {


        //AsyncTaskRunnerepondre runner2 = new AsyncTaskRunnerepondre(UserActivity.this);

     //   runner2.execute();


    }

});*/
        imageViewResult.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {


               OpenImages();


            }

        });
    }

    public boolean checkPermission(){

        int result = ContextCompat.checkSelfPermission(UserActivity.this, Manifest.permission.READ_EXTERNAL_STORAGE);
        int result1 = ContextCompat.checkSelfPermission(UserActivity.this,Manifest.permission.WRITE_EXTERNAL_STORAGE);
        int result2 = ContextCompat.checkSelfPermission(UserActivity.this,Manifest.permission.CAMERA);

        return result == PackageManager.PERMISSION_GRANTED && result1 == PackageManager.PERMISSION_GRANTED && result2 == PackageManager.PERMISSION_GRANTED;

    }

    public void requestPermission(){

        ActivityCompat.requestPermissions(UserActivity.this,new String[] {Manifest.permission.READ_EXTERNAL_STORAGE,Manifest.permission.WRITE_EXTERNAL_STORAGE,Manifest.permission.CAMERA},PERMISSION_CODE);

    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {

        switch (requestCode){

            case PERMISSION_CODE :

                if (grantResults.length > 0){

                    boolean storage  = grantResults[0] == PackageManager.PERMISSION_GRANTED;
                    boolean cameras = grantResults[0] == PackageManager.PERMISSION_GRANTED;

                    if (storage && cameras){

                        Toast.makeText(UserActivity.this, "Permission Granted", Toast.LENGTH_SHORT).show();

                    }else{

                        Toast.makeText(UserActivity.this, "Permission Denied", Toast.LENGTH_SHORT).show();

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

        new AlertDialog.Builder(UserActivity.this)
                .setMessage(s)
                .setPositiveButton("OK", listener)
                .setNegativeButton("Cancel", null)
                .create()
                .show();
    }

    public void OpenImages() {

        final CharSequence[] option = {"Camera","Gellary"};

        AlertDialog.Builder builder = new AlertDialog.Builder(UserActivity.this);
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

        imageViewResult.setImageBitmap(bitmap);
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
        bitmap = (Bitmap) data.getExtras().get("data");
        int orientation = getOrientationOfImage(path);

        bitmap = getRotatedBitmap(bitmap, orientation);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();

        bitmap.compress(Bitmap.CompressFormat.PNG,90,bytes);

        File paths = new File(Environment.getExternalStorageDirectory(), System.currentTimeMillis() + ".png");
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

        } catch (IOException e) {
            e.printStackTrace();
        }
        textView.setText("Source image ");
        textView.setTextColor(Color.BLACK);
        imageViewResult.setImageBitmap(bitmap);
        img_input = new Mat();
        Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        Utils.bitmapToMat(bmp32, img_input);

    }
/*

    public List<couleur1>   RecognitionColor(){


        Mat decol;
        decol = new Mat();Mat can1;
        can1=new Mat();distancecouleur = new ArrayList<couleur1>();

        MainActivity7.histths(img_output.getNativeObjAddr(),decol.getNativeObjAddr());

        Cursor cursor = MainActivity7.sqLiteHelper.getData("SELECT * FROM COLORF  ");

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
                couleur1 foo = new couleur1(d, s1);
                distancecouleur.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();

        return distancecouleur;
    }
    List<huu1>  Recognitionhuu(){


        Mat out = new Mat();
        distancehu = new ArrayList<huu1>();
        Bitmap bitmapOutput2 = null;
        // excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        String HUvct="";
        double [] h1= MainActivity7.humom(img_output.getNativeObjAddr(),out.getNativeObjAddr());


        Cursor cursor = MainActivity7.sqLiteHelper.getData("SELECT * FROM HUMOMO  ");

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
            huu1 foo3= new huu1(d, s1);
            distancehu.add(foo3);
             }
        cursor.close();


        return distancehu;
    }
    List<llbp1>   RecognitionELbp(){


        Mat delbp;
        delbp = new Mat();
        Mat hist=new Mat();
        lbpinput = new Mat();
        //excract(img_input.getNativeObjAddr(),  lbpinput.getNativeObjAddr());
        Bitmap bitmapOutput5 =null;
        MainActivity7.elbbp(img_output.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
        distancelbp = new ArrayList<llbp1>();
        Cursor cursor = MainActivity7.sqLiteHelper.getData("SELECT * FROM LBP  ");

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
                llbp1 foo = new llbp1(d, s1);
                distancelbp.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();

        return distancelbp;
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

    String distanceponde(List<couleur1> a, List<huu1> b, List<llbp1> c) {
        double distance;
        String nom;
        String dis="";
        distancefinal = new ArrayList<distancepobdre1>();
        for (int i = 0; i < a.size();i++) {
            nom=a.get(i).namobject;
            distance = a.get(i).distance+c.get(i).distance+b.get(i).distance;
            distancepobdre1 ff = new distancepobdre1(distance, nom);
            distancefinal.add(ff);
        }
        Collections.sort(distancefinal, new Sortbyroll41());
        Log.v("shit","***********************************************************************");
        for (distancepobdre1 d: distancefinal) {
            dis+=d.namobject+" , ";
            // Log.v("Array Value","Array Value"+d.namobject+" "+d.distance);
        }
        String fdata[] = dis.split(" , ");
        String res="";
        res+=fdata[0]+" , "+fdata[1]+" , "+fdata[2]+" , "+fdata[3] ;
        return  res;
    }

    private class AsyncTaskRunnerepondre extends AsyncTask<String, String, String> {

        private String resp;
        private ProgressDialog dialog;
        public AsyncTaskRunnerepondre(UserActivity activity) {
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
            MainActivity7.excract(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
            t= new UserActivity.Thread0();
            t1 = new UserActivity.Thread1();
            t2 = new UserActivity.Thread2();
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

            Intent intent = new Intent(UserActivity.this, ReActivity.class);
            intent.putExtra("recognition", result);
            // intent.putExtra("Image", bitmap);
            intent.putExtra("imgurl", u);
            startActivity(intent);

        }
    }

*/
}