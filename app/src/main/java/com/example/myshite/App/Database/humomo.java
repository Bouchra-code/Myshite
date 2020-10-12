package com.example.myshite.App.Database;

public class humomo {
    private int id;
    private String name;

    private String huvector;
    public  humomo (String name, String huvector, int id) {
        this.name = name;
        this.huvector = huvector;
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

    public String getHuvector() {
        return huvector;
    }
    public void setHuvector(String huvector) {
        this.huvector = huvector;
    }
}
