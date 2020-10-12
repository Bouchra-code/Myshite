package com.example.myshite;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.myshite.App.Activitycheby;
import com.example.myshite.App.Database.ObjectList;
import com.example.myshite.App.InsertOb;
import com.example.myshite.App.UserActivity;
import com.google.android.material.bottomsheet.BottomSheetDialog;

import java.io.IOException;

public class ReActivity extends AppCompatActivity {
    TextView textView, textView1;
    private TextView textViewResult;
    private static final int INPUT_SIZE = 224;
    private ImageView imageViewResult;
    private Toolbar toolbar;
    private Button buttonShow;
    private static String path1;
    String res1="";
    String res2="";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_re);
      //  textViewResult = findViewById(R.id.textView2);
        textView = findViewById(R.id.text);
        textView1 = findViewById(R.id.text1);
        imageViewResult = findViewById(R.id.imageView8);

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);




        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setHomeButtonEnabled(true);
        getSupportActionBar().setTitle("");
        Intent intent = getIntent();
        String string2 = intent.getStringExtra("recognition");
        Bundle extras = getIntent().getExtras();

        // byte[] byteArray = extras.getByteArray("picture");

        // Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

        Uri path = (Uri) extras.get("imgurl");


        final String fdata[] = string2.split(" , ");

        imageViewResult.setImageURI(path);
       // imageViewResult.setImageBitmap(bitmap);
        textView.setText("Source image ");
        textView.setTextColor(Color.BLACK);
        //  Bitmap bitmap1 = (Bitmap) intent.getParcelableExtra("Image");
        buttonShow=findViewById(R.id.buttons);

        if(fdata[0].startsWith("apple")) {
            res1 = "apple";

        }else  if(fdata[0].startsWith("pear")){
            res1="pear";
        }else  if(fdata[0].startsWith("cup")){
            res1="cup";
        }
       else  if(fdata[0].startsWith("car")){
            res1="car";
        }else  if(fdata[0].startsWith("cow")){
            res1="cow";
        }else  if(fdata[0].startsWith("horse")){
            res1="horse";
        }else  if(fdata[0].startsWith("dog")){
            res1="dog";
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
        }else if(fdata[0].startsWith("tomat")){
            res1="tomato";
        }else if(fdata[0].startsWith("pome")){
            res1="pomegranate";
        }else if(fdata[0].startsWith("lemo")){
            res1="lemon";
        }
        buttonShow.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
// here you can bind bottom sheet dialog and initialize Theme in bottom sheet dialog
                final BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(ReActivity.this,
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
                        Intent intent = new Intent(ReActivity.this, ObjectList.class);
                        intent.putExtra("fdata0", fdata[0]);
                        // intent.putExtra("Image", bitmap);
                        //  intent.putExtra("picture", byteArray);
                        intent.putExtra("fdata1", fdata[1]);
                        startActivity(intent);
                    }
                });
                bottomSheetDialog.setContentView(bottomSheetView);
                bottomSheetDialog.show();

            }
        });


        //textView1.setText("Recognition results ");
        //textView1.setTextColor(Color.BLACK);

    }  private void rotateimage(Bitmap bitmap) throws IOException {
        ExifInterface exifInterface=null;
        exifInterface =new ExifInterface(path1);
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
}