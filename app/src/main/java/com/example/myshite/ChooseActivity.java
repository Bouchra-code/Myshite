package com.example.myshite;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;

import com.example.myshite.App.Activitycheby;
import com.example.myshite.App.ChooseimageActivity;
import com.example.myshite.App.UserActivity;

public class ChooseActivity extends AppCompatActivity {
private CardView usercard,prodcard;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_choose);
       usercard=findViewById(R.id.user);
       prodcard=findViewById(R.id.prof);


        usercard.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {

                Intent intent = new Intent(ChooseActivity.this, ChooseimageActivity.class);

                startActivity(intent);
            }

        });
        prodcard.setOnClickListener(new View.OnClickListener() {

            @Override

            public void onClick(View v) {

                Intent intent = new Intent(ChooseActivity.this, MainActivity7.class);

                startActivity(intent);
            }

        });
    }

}