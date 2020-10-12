package com.example.myshite.App;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;


import com.example.myshite.App.Database.ObjectList;
import com.example.myshite.R;
import com.example.myshite.ReActivity;
import com.google.android.material.bottomsheet.BottomSheetDialog;

public class Activitycheby extends AppCompatActivity {

    TextView textView, textView1;
    private TextView textViewResult;
    private static final int INPUT_SIZE = 224;
    private ImageView imageViewResult;
    private Toolbar toolbar;
    private Button buttonShow;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_activitycheby);
//        getActionBar().setDisplayHomeAsUpEnabled(true);

        textView = findViewById(R.id.text);

        imageViewResult = findViewById(R.id.imageView8);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        getSupportActionBar().setDisplayHomeAsUpEnabled(true);
        getSupportActionBar().setHomeButtonEnabled(true);
        getSupportActionBar().setTitle("");
        Intent intent = getIntent();
        String string2 = intent.getStringExtra("chebymoment");
        final String fdata[] = string2.split(" , ");
        //  Bitmap bitmap1 = (Bitmap) intent.getParcelableExtra("Image");

        Bundle extras = getIntent().getExtras();
       // byte[] byteArray = extras.getByteArray("picture");

       // Bitmap bmp = BitmapFactory.decodeByteArray(byteArray, 0, byteArray.length);

            Uri path = (Uri) extras.get("imgurl");

        imageViewResult.setImageURI(path);
        textView.setText("Source image ");
        textView.setTextColor(Color.BLACK);

      //  textViewResult.setTextColor(Color.BLACK);
      //textViewResult.setText("\t"+"\t"+"\t"+"\t"+"T03 = "+fdata[0]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T12 = "+fdata[1]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T23 = "+fdata[2]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T30 = "+fdata[3]+"\n"+"\n");
        buttonShow=findViewById(R.id.buttons);
        buttonShow.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
// here you can bind bottom sheet dialog and initialize Theme in bottom sheet dialog
                final BottomSheetDialog bottomSheetDialog = new BottomSheetDialog(Activitycheby.this,
                        R.style.BottomSheetDialogTheme);
// here you can inflate layout which will be shows in bottom sheet dialog
                View bottomSheetView = LayoutInflater.from(getApplicationContext())
                        .inflate(R.layout.bottomsheet,
                                (LinearLayout)findViewById(R.id.bottom_sheet_container));
                TextView textreco = (TextView) bottomSheetView.findViewById(R.id.reco);
                textreco.setTextColor(Color.BLACK);
                textreco.setText("\t"+"\t"+"\t"+"\t"+"T03 = "+fdata[0]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T12 = "+fdata[1]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T23 = "+fdata[2]+"\n"+"\n"+"\t"+"\t"+"\t"+"\t"+"T30 = "+fdata[3]+"\n"+"\n");

                bottomSheetDialog.setContentView(bottomSheetView);
                bottomSheetDialog.show();

            }
        });



    }


    


}