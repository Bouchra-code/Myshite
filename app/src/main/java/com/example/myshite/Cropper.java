package com.example.myshite;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;

import com.theartofdev.edmodo.cropper.CropImage;

public class Cropper extends AppCompatActivity {
    ImageView imag;

    Uri mImageUri;



    @Override

    protected void onCreate(@Nullable Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_cropper);



        imag = findViewById(R.id.imageView4);

        imag.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {

                CropImage.activity().start(Cropper.this);

            }

        });



    }



    @Override

    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        super.onActivityResult(requestCode, resultCode, data);



        if(requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE){





            CropImage.ActivityResult result = CropImage.getActivityResult(data);

            if(resultCode ==RESULT_OK){

                mImageUri = result.getUri();

                imag.setImageURI(mImageUri);



            }else if(resultCode == CropImage.CROP_IMAGE_ACTIVITY_RESULT_ERROR_CODE){

                Exception e = result.getError();

                Log.d("error",e.toString());

            }

        }

    }

}