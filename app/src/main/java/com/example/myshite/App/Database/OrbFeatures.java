package com.example.myshite.App.Database;

public class OrbFeatures {


    private int id;
    private String name;
    int t;
    int w;
    int h;

    private byte[] image;

    public OrbFeatures(String name, byte[] image, int id) {
        this.name = name;

        this.image = image;
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public int getT() {
        return t;
    }

    public int getW() {
        return w;
    }

    public void setT(int t) {
        this.t = t;
    }

    public void setH(int h) {
        this.h = h;
    }

    public void setW(int w) {
        this.w = w;
    }

    public int getH() {
        return h;
    }

    public void setId(int id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }


    public byte[] getImage() {
        return image;
    }

    public void setImage(byte[] image) {
        this.image = image;
    }
}
