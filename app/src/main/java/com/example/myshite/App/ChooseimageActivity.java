package com.example.myshite.App;

import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;


import com.example.myshite.App.Database.ObjectList;
import com.example.myshite.App.Database.SQLiteHelper;
import com.example.myshite.MainActivity7;
import com.example.myshite.R;
import com.google.android.material.bottomsheet.BottomSheetDialog;


import android.app.Activity;
import android.app.ProgressDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
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

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Date;
import java.util.List;
import java.util.Locale;

import static com.example.myshite.MainActivity7.calculateDistance;
import static org.opencv.core.CvType.CV_32F;


class llbp1 {
    public double distance;
    public String namobject;
    public llbp1(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}

class couleur1{
    public double distance;
    public String namobject;
    public couleur1(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}

class huu1 {
    public double distance;
    public String namobject;
    public huu1(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class distancepobdre1 {
    public double distance;
    public String namobject;
    public distancepobdre1(  double distance, String namobject){
        this.distance =distance;
        this.namobject = namobject;

    }
}
class Sortbyroll41 implements Comparator<distancepobdre1>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(distancepobdre1 a, distancepobdre1 b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyrollhuu1 implements Comparator<huu1>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(huu1 a, huu1 b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll11 implements Comparator<llbp1>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(llbp1 a, llbp1 b)
    {
        return Double.compare(a.distance, b.distance);
    }
}
class Sortbyroll21 implements Comparator<couleur1>
{
    // Used for sorting in ascending order of
    // roll number
    public int compare(couleur1 a, couleur1 b)
    {
        return Double.compare(a.distance, b.distance);
    }
}


public class ChooseimageActivity extends AppCompatActivity {
    double w3=0.6355;//couleur
    double w1=0.3052;//chebychev
    double w2=0.5842;//elbp
    Thread t=null;
    Thread t1=null;
    Thread t2=null;
    private static String path;
    private CardView card1,card2;
    Intent intent;
    Uri fileUri;
    String res1="";
    Button btn_choose_image;
    ImageView imageView;
    Bitmap bitmap, decoded;
    SQLiteHelper db;
    public final int REQUEST_CAMERA = 0;
    public final int SELECT_FILE = 1;
    private Mat img_output, out, filtinput, canyinput, orbinput, lbpinput;
    private Mat img_input, matdegris, matdecanny, matdeorb, matdelbp, matfilter;
String res="";
    byte[] byteArray;
    Uri u ;
    List<couleur1> distancecouleur ;

    Button buttonShow;
    List<huu1> distancehu ;
    List<llbp1> distancelbp ;
    List<distancepobdre1> distancefinal ;
    int bitmap_size = 40; // image quality 1 - 100;
    int max_resolution_image = 800;
    Button recognition ;
    static {
        System.loadLibrary("opencv_java3");
        System.loadLibrary("native-lib");
    }
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_chooseimage);

imageView=findViewById(R.id.imgg);
recognition=findViewById(R.id.buttons);
        card2=findViewById(R.id.galerie);
        final String fdata[] = res.split(" , ");

        card2.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {
                selectImage();

            }

        });

        recognition.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {


              AsyncTaskRunnerepon runner2 = new AsyncTaskRunnerepon(ChooseimageActivity.this);

                runner2.execute();
// here you can bind bottom sheet dialog and initialize Theme in bottom sheet dialog

             /*   t= new Thread0();
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

                final String fdata[] = s1.split(" , ");
                if(fdata[0].startsWith("apple")) {
                    res1 = "apple";

                }
                else  if(fdata[0].startsWith("car")){
                    res1="car";
                }
                else if(fdata[0].startsWith("pen")){
                    res1="pen";


                }else if(fdata[0].startsWith("onion")){
                    res1="onion";
                }else if(fdata[0].startsWith("garlic")){
                    res1="garlic";
                }else if(fdata[0].startsWith("bott")){
                    res1="bottle";
                }else if(fdata[0].startsWith("cucu")){
                    res1="cucumber";

                }else
                if(fdata[0].startsWith("cucu")){
                    res1="cucumber";
                }else if(fdata[0].startsWith("zucc")){
                    res1="zucchini";
                }else if(fdata[0].startsWith("zucc")){
                    res1="zucchini";
                }*/
               /* final BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(ChooseimageActivity.this,
                        R.style.BottomSheetDialogTheme);
// here you can inflate layout which will be shows in bottom sheet dialog
                View bottomSheetView = LayoutInflater.from(getApplicationContext())
                        .inflate(R.layout.bottom_sheet_dialog,
                                (LinearLayout)findViewById(R.id.bottom_sheet_container));
                TextView textreco = (TextView) bottomSheetView.findViewById(R.id.reco);
                textreco.setTextColor(Color.BLACK);
                textreco.setText("\t"+"\t"+"\t"+"\t"+res1+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"\n"+"\n");


                bottomSheetView.findViewById(R.id.button_share).setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        Intent intent = new Intent(ChooseimageActivity.this, ObjectList.class);
                        intent.putExtra("fdata0", fdata[0]);
                        // intent.putExtra("Image", bitmap);
                        //  intent.putExtra("picture", byteArray);
                        intent.putExtra("fdata1", fdata[1]);
                        startActivity(intent);
                    }
                });
                bottomSheetDialog.setContentView(bottomSheetView);
                bottomSheetDialog.show();
*/
            }
        });

    }
    private void selectImage() {
        imageView.setImageResource(0);
        final CharSequence[] items = {"Take Photo", "Choose from Library",
                "Cancel"};

        AlertDialog.Builder builder = new AlertDialog.Builder(ChooseimageActivity.this);
        builder.setTitle("Add Photo!");
        builder.setIcon(R.mipmap.ic_launcher);
        builder.setItems(items, new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int item) {
                if (items[item].equals("Take Photo")) {
                    intent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
                    fileUri = getOutputMediaFileUri();
                    intent.putExtra(android.provider.MediaStore.EXTRA_OUTPUT, fileUri);
                    startActivityForResult(intent, REQUEST_CAMERA);
                } else if (items[item].equals("Choose from Library")) {
                    intent = new Intent();
                    intent.setType("image/*");
                    intent.setAction(Intent.ACTION_GET_CONTENT);
                    startActivityForResult(Intent.createChooser(intent, "Select Picture"), SELECT_FILE);
                } else if (items[item].equals("Cancel")) {
                    dialog.dismiss();
                }
            }
        });
        builder.show();
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        Log.e("onActivityResult", "requestCode " + requestCode + ", resultCode " + resultCode);

        if (resultCode == Activity.RESULT_OK) {
            if (requestCode == REQUEST_CAMERA) {
                try {
                    Log.e("CAMERA", fileUri.getPath());

                    bitmap = BitmapFactory.decodeFile(fileUri.getPath());
                  rotateimage(getResizedBitmap(bitmap, max_resolution_image));
                    img_input = new Mat();
                    Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Utils.bitmapToMat(bmp32, img_input);
                    //setToImageView();
                } catch (Exception e) {
                    e.printStackTrace();
                }
            } else if (requestCode == SELECT_FILE && data != null && data.getData() != null) {
                try {
                    // mengambil gambar dari Gallery
                    bitmap = MediaStore.Images.Media.getBitmap(ChooseimageActivity.this.getContentResolver(), data.getData());
                    //setToImageView(getResizedBitmap(bitmap, max_resolution_image));
                    img_input = new Mat();
                    Bitmap bmp32 = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    Utils.bitmapToMat(bmp32, img_input);
                   setToImageView(getResizedBitmap(bitmap, max_resolution_image));

                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }

    // Untuk menampilkan bitmap pada ImageView
    private void setToImageView(Bitmap bmp) {
        //compress image
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bmp.compress(Bitmap.CompressFormat.JPEG, bitmap_size, bytes);
        decoded = BitmapFactory.decodeStream(new ByteArrayInputStream(bytes.toByteArray()));

        //menampilkan gambar yang dipilih dari camera/gallery ke ImageView
        imageView.setImageBitmap(decoded);
    }

    // Untuk resize bitmap
    public Bitmap getResizedBitmap(Bitmap image, int maxSize) {
        int width = image.getWidth();
        int height = image.getHeight();

        float bitmapRatio = (float) width / (float) height;
        if (bitmapRatio > 1) {
            width = maxSize;
            height = (int) (width / bitmapRatio);
        } else {
            height = maxSize;
            width = (int) (height * bitmapRatio);
        }
        return Bitmap.createScaledBitmap(image, width, height, true);
    }

    public Uri getOutputMediaFileUri() {
        return Uri.fromFile(getOutputMediaFile());
    }

    private static File getOutputMediaFile() {

        // External sdcard location
        File mediaStorageDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "DeKa");

        // Create the storage directory if it does not exist
        if (!mediaStorageDir.exists()) {
            if (!mediaStorageDir.mkdirs()) {
                Log.e("Monitoring", "Oops! Failed create Monitoring directory");
                return null;
            }
        }

        // Create a media file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(new Date());
        File mediaFile;
        mediaFile = new File(mediaStorageDir.getPath() + File.separator + "IMG_DeKa_" + timeStamp + ".jpg");
path=mediaFile.getAbsolutePath();
        return mediaFile;
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
        imageView.setImageBitmap(rptatedimage);
        }
//////////////////////////////////////////////////////////////////////////////////
    public native void excract1(long inputImage, long outputImage);
    public native void elbbp1(long inputImage, long outputImage,long hist);
    public native void histths1(long inputImage, long outputImage);
    public native double[] humom1(long inputImage, long outputImage);
    ///////////////////////////////////////////////////////////////
    List<llbp1>   RecognitionELbp(){


        Mat delbp;
        delbp = new Mat();
        Mat hist=new Mat();
        lbpinput = new Mat();
        //excract1(img_input.getNativeObjAddr(),  lbpinput.getNativeObjAddr());
        Bitmap bitmapOutput5 =null;
        elbbp1(lbpinput.getNativeObjAddr(),  delbp.getNativeObjAddr(),hist.getNativeObjAddr());
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

                double res = Imgproc.compareHist(hist,bb, Imgproc.CV_COMP_CHISQR);
                Double d = new Double(res * 100);
                String s1 = cursor.getString(1);
                llbp1 foo = new llbp1(d, s1);
                distancelbp.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();
        /*Collections.sort(distancelbp, new Sortbyroll11());



        for (llbp1 a: distancelbp) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }*/
        return distancelbp;
    }
    List<couleur1> RecognitionColor(){


        Mat decol;
        img_output=new Mat();
        decol = new Mat();Mat can1;
        //excract1(img_input.getNativeObjAddr(),  img_output.getNativeObjAddr());
        histths1(img_output.getNativeObjAddr(),decol.getNativeObjAddr());

        Cursor cursor =MainActivity7.sqLiteHelper.getData("SELECT * FROM COLORF  ");

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
                double d = Imgproc.compareHist(decol,bb, Imgproc.CV_COMP_BHATTACHARYYA);

                String s1 = cursor.getString(1);
                couleur1 foo = new couleur1(d, s1);
                distancecouleur.add(foo);
                cursor.moveToNext();
            }
        }cursor.close();

      /*  Collections.sort(distancecouleur, new Sortbyroll21());
       for (couleur1 a: distancecouleur) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }
*/
        return distancecouleur;
    }


    List<huu1>  Recognitionhuu(){


        // img_output = new Mat();
        Mat bb1 = new Mat();
        Mat out = new Mat();
        img_output=new Mat();
        distancehu = new ArrayList<huu1>();
        Bitmap bitmapOutput2 = null;
         //excract1(img_input.getNativeObjAddr(), img_output.getNativeObjAddr());
        String HUvct="";
        double [] h1= humom1(img_output.getNativeObjAddr(),out.getNativeObjAddr());


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
            // cursor.moveToNext();


        }
        cursor.close();
        /*Collections.sort(distancehu, new Sortbyrollhuu1());
        for (huu1 a: distancehu) {

            Log.v("Array Value","Array Value"+a.namobject+" "+a.distance);
        }*/
        return distancehu;
    }

    class Thread0 extends Thread {

        @Override
        public void run() {
            //Log.e("thread 0 B", "herer");
            distancecouleur=RecognitionColor();
            //  Log.e("thread 0 RES", "herer");
        }

    }
    class Thread1 extends Thread {

        @Override
        public void run() {
            //Log.e("thread 1 B", "herer");
            distancelbp=RecognitionELbp();
            //Log.e("thread 1 RES", "herer");
        }
    }
    class Thread2 extends Thread {

        @Override
        public void run() {
            // Log.e("thread 2 B", "herer");
            distancehu= Recognitionhuu();
            //Log.e("thread 2 RES", "herer");
        }

    }

/*************************************/
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
    Log.v("shit","**********************************");
    for (distancepobdre1 d: distancefinal) {
        dis+=d.namobject+" , ";
        // Log.v("Array Value","Array Value"+d.namobject+" "+d.distance);
    }
    String fdata[] = dis.split(" , ");
    String res="";
    res+=fdata[0]+" , "+fdata[1]+" , "+fdata[2]+" , "+fdata[3] ;
    return  res;
}


    /****************************************************************/
    private class AsyncTaskRunnerepon extends AsyncTask<String, String, String> {

        private String resp;
        private ProgressDialog dialog;
        public AsyncTaskRunnerepon(ChooseimageActivity activity) {
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
            img_output = new Mat();
            excract1(img_input.getNativeObjAddr(),  img_output.getNativeObjAddr());
            t= new ChooseimageActivity.Thread0();
            t1 = new ChooseimageActivity.Thread1();
            t2 = new ChooseimageActivity.Thread2();
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

            double distance;
            String nom;
            String dis="";
          distancefinal = new ArrayList<distancepobdre1>();
            for (int i = 0; i < distancecouleur.size();i++) {
                nom=distancecouleur.get(i).namobject;
                distance = w3*distancecouleur.get(i).distance+w1*distancehu.get(i).distance+w2*distancelbp.get(i).distance;
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


        @Override
        protected void onPostExecute(String result) {
            // This method is executed in the UIThread
            // with access to the result of the long running task
            if (dialog.isShowing()) {
                dialog.dismiss();
            }
res=result;
           /* Intent intent = new Intent(MainActivity7.this, ReActivity.class);
            intent.putExtra("recognition", result);
            // intent.putExtra("Image", bitmap);
            intent.putExtra("imgurl", u);
            startActivity(intent);*/

        }
    }







}



