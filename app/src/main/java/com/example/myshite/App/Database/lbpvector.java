package com.example.myshite.App.Database;

public class lbpvector {

    private int id;
    private String name;
    private int size ;
    private int[] vectorfeature;
    public int getId() {
        return id;
    }

    public void  setId() {
        this.id = id;
    }
    public int getSize() {
        return size;
    }

    public void  setSize() {
        this.size = size;
    }
    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
    public int[] getImage() {
        return vectorfeature;
    }

    public void setImage(int[] image) {
        this.vectorfeature = image;
    }
}
