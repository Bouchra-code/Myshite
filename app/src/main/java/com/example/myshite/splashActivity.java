package com.example.myshite;

import android.content.Intent;
import android.os.Bundle;
import android.view.View;
import android.view.WindowManager;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.cardview.widget.CardView;

import com.example.myshite.App.InsertOb;
import com.example.myshite.App.RecoActivity;

public class splashActivity extends AppCompatActivity {
    private final int SPLASH_DISPLAY_LENGTH=8000;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash);
        CardView cardView= findViewById(R.id.cardview);
        // CardView cardView1=findViewById(R.id.cardview1);
        TextView text=findViewById(R.id.text);
        TextView text1=findViewById(R.id.text1);
        TextView text2=findViewById(R.id.text2);
        TextView text3=findViewById(R.id.text3);
        ImageView imageView=findViewById(R.id.imageView);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,WindowManager.LayoutParams.FLAG_FULLSCREEN);
        Animation animation, animation2,animationWelcome,animobject,animRec,animphone,cursiveText;
        animation= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.fade_in);
        animation2= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.bounce);
        animationWelcome= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.squential);
        animobject= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.open_next);
        animRec= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.close_next);
        animphone= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.bounce);
        cursiveText= AnimationUtils.loadAnimation(getApplicationContext(),R.anim.open_next);
        //cardView1.startAnimation(animation);
        cardView.startAnimation(animation2);
        text3.startAnimation(animationWelcome);
        text.startAnimation(animobject);
        text1.startAnimation(animRec);
        imageView.startAnimation(animphone);
        text2.startAnimation(cursiveText);
        /*new Handler().postDelayed(new Runnable() {
            @Override
            public void run() {
                Intent intent=new Intent(splashActivity.this,MainActivity.class);
                splashActivity.this.startActivity(intent);
                splashActivity.this.finish();

            }
        },SPLASH_DISPLAY_LENGTH);*/
        cardView.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(splashActivity.this, ChooseActivity.class);
                startActivity(intent);
                splashActivity.this.finish();

            }



        });


    }
}