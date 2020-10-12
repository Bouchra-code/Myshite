package com.example.myshite.App.Database;

public class cheychevseven {
    private int id;
    private String name;

    private String chebyvector;
    public  cheychevseven (String name, String chebyvector, int id) {
        this.name = name;
        this.chebyvector = chebyvector;
        this.id = id;

    }

    public int getId() {
        return id;
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

    public String getChebyvector() {
        return chebyvector;
    }
    public void setChebyvector(String chebyvector) {
        this.chebyvector = chebyvector;
    }

}
