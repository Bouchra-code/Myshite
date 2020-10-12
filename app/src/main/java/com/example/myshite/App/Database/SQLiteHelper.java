package com.example.myshite.App.Database;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.sqlite.SQLiteStatement;
import android.util.Log;

import com.example.myshite.App.InsertOb;

import org.opencv.core.Mat;

import java.util.ArrayList;

import static com.example.myshite.App.InsertOb.sqLiteHelper;

public class SQLiteHelper extends SQLiteOpenHelper {

    public SQLiteHelper(Context context, String name, SQLiteDatabase.CursorFactory factory, int version) {
        super(context, name, factory, version);
    }

    public void queryData(String sql){
        SQLiteDatabase database = getWritableDatabase();
        database.execSQL(sql);
    }

    public void insertData(String name, byte[] image){
        SQLiteDatabase database = getWritableDatabase();
        String sql = "INSERT INTO OBJECT VALUES (NULL, ?, ?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);

        statement.bindBlob(2, image);

        statement.executeInsert();
    }

    public  void deleteData(int id) {
        SQLiteDatabase database = getWritableDatabase();

        String sql = "DELETE FROM OBJECT WHERE id = ?";
        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();
        statement.bindDouble(1, (double)id);

        statement.execute();
        database.close();
    }

    public void updateData(String name,  byte[] image, int id) {
        SQLiteDatabase database = getWritableDatabase();

        String sql = "UPDATE OBJECT SET name = ?, image = ? WHERE id = ?";
        SQLiteStatement statement = database.compileStatement(sql);

        statement.bindString(1, name);

        statement.bindBlob(2, image);


        statement.execute();
        database.close();
    }


    //sqLiteHelper.queryData("CREATE TABLE IF NOT EXISTS LBP(Id INTEGER PRIMARY KEY AUTOINCREMENT, name VARCHAR, image BLOB)");

    public Cursor getData(String sql){
        SQLiteDatabase database = getReadableDatabase();
        return database.rawQuery(sql, null);
    }
    public void dbput1(String name, Mat m) {
        long nbytes = m.total() * m.elemSize();
        byte[] bytes = new byte[ (int)nbytes ];
        m.get(0, 0,bytes);

        dbput1(name, m.type(), m.cols(), m.rows(), bytes);
    }

    public void dbput1(String name, int t, int w, int h, byte[] bytes) {

        SQLiteDatabase database  = this.getWritableDatabase();
        String sql = "INSERT INTO  ORBFEATURE VALUES (NULL, ?, ?,?,?,?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);
        statement.bindLong(2,t);

        statement.bindLong(3,w);
        statement.bindLong(4,h);
        statement.bindBlob(5,bytes);
        statement.executeInsert();
    }

    public Mat dbget1(String name) {
        SQLiteDatabase db = this.getReadableDatabase();
        String [] columns = {"t","w","h","pix"};
        Cursor cursor = this.getData("SELECT * FROM ORBFEATURE");

        if (cursor != null)
            cursor.moveToFirst();

        int t = cursor.getInt(2);
        int w = cursor.getInt(3);
        int h = cursor.getInt(4);
        byte[] p = cursor.getBlob(5);
        Mat m = new Mat(h,w,t);
        m.put(0,0,p);
        // Log.d("dbget("+name+")", m.toString());
        return m;
    }


    public void dbput(String name, Mat m) {
        long nbytes = m.total() * m.elemSize();
        byte[] bytes = new byte[ (int)nbytes ];
        m.get(0, 0,bytes);

        dbput(name, m.type(), m.cols(), m.rows(), bytes);
    }

    public void dbput(String name, int t, int w, int h, byte[] bytes) {
        Log.d("dbput", name + " " + t + " " + w + "x" + h);
        SQLiteDatabase database  = this.getWritableDatabase();
        String sql = "INSERT INTO  LBP VALUES (NULL,?,?,?,?,?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);
        statement.bindLong(2,t);

        statement.bindLong(3,w);
        statement.bindLong(4,h);
        statement.bindBlob(5,bytes);
        statement.executeInsert();
    }

    public Mat dbget(String name) {
        SQLiteDatabase db = this.getReadableDatabase();
        String [] columns = {"t","w","h","pix"};
        Cursor cursor = this.getData("SELECT * FROM LBP");

        if (cursor != null)
            cursor.moveToFirst();

        int t = cursor.getInt(2);
        int w = cursor.getInt(3);
        int h = cursor.getInt(4);
        byte[] p = cursor.getBlob(5);
        Mat m = new Mat(h,w,t);
        m.put(0,0,p);
        // Log.d("dbget("+name+")", m.toString());
        return m;
    }
   /* public void Insertfea(String name, int size,  int [] bytes) {

        SQLiteDatabase database  = this.getWritableDatabase();
        String sql = "INSERT INTO  LbpFEATURE VALUES (NULL, ?, ?,?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);
        statement.bindLong(2,size);
        statement.bindBlob();
        statement.executeInsert();
    }*/


    public void dbputshape(String name, Mat m) {
        long nbytes = m.total() * m.elemSize();
        byte[] bytes = new byte[ (int)nbytes ];
        m.get(0, 0,bytes);

        dbputshape(name, m.type(), m.cols(), m.rows(), bytes);
    }

    public void dbputshape(String name, int t, int w, int h, byte[] bytes) {
        Log.d("dbput", name + " " + t + " " + w + "x" + h);
        SQLiteDatabase database  = this.getWritableDatabase();
        String sql = "INSERT INTO  SHAPE VALUES (NULL,?,?,?,?,?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);
        statement.bindLong(2,t);

        statement.bindLong(3,w);
        statement.bindLong(4,h);
        statement.bindBlob(5,bytes);
        statement.executeInsert();
    }
    public void  getImage(String nameE,  ArrayList<Object> list) {
        SQLiteDatabase db = this.getReadableDatabase();

        Cursor cursor = db.rawQuery("SELECT * FROM OBJECT WHERE name = ?", new String[]{nameE});
        //list.clear();
        if( cursor != null && cursor.moveToFirst() ) {
            int id = cursor.getInt(0);
            String name = cursor.getString(1);
            byte[] image = cursor.getBlob(2);

            list.add(new Object(name, image, id));
            cursor.close();
        }

    }
    public Mat dbgetshape(String name) {
        SQLiteDatabase db = this.getReadableDatabase();
        String [] columns = {"t","w","h","pix"};
        Cursor cursor = this.getData("SELECT * FROM SHAPE ");

        if (cursor != null)
            cursor.moveToFirst();

        int t = cursor.getInt(2);
        int w = cursor.getInt(3);
        int h = cursor.getInt(4);
        byte[] p = cursor.getBlob(5);
        Mat m = new Mat(h,w,t);
        m.put(0,0,p);
        // Log.d("dbget("+name+")", m.toString());
        return m;
    }
    public void insertcheby(String name, String chebybector){
        SQLiteDatabase database = getWritableDatabase();
        String sql = "INSERT INTO CHEBY VALUES (NULL, ?, ?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);

        statement.bindString(2, chebybector);

        statement.executeInsert();
    }
    public void inserthu(String name, String huvector){
        SQLiteDatabase database = this.getWritableDatabase();
        String sql = "INSERT INTO HU VALUES (NULL, ?, ?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);

        statement.bindString(2, huvector);

        statement.executeInsert();
    }
    public void inserthu1(String name, String huvector){
        SQLiteDatabase database = this.getWritableDatabase();
        String sql = "INSERT INTO HUMOMO VALUES (NULL, ?, ?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);

        statement.bindString(2, huvector);

        statement.executeInsert();
    }
    public void insertchebyseven(String name, String chebybector){
        SQLiteDatabase database = getWritableDatabase();
        String sql = "INSERT INTO CHEBYNINE VALUES (NULL, ?, ?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);

        statement.bindString(2, chebybector);

        statement.executeInsert();
    }
    public void dbputcolor(String name, Mat m) {
        long nbytes = m.total() * m.elemSize();
        byte[] bytes = new byte[ (int)nbytes ];
        m.get(0, 0,bytes);

        dbputcolor(name, m.type(), m.cols(), m.rows(), bytes);
    }

    public void dbputcolor(String name, int t, int w, int h, byte[] bytes) {

        SQLiteDatabase database  = this.getWritableDatabase();
        String sql = "INSERT INTO  COLORF VALUES (NULL, ?, ?,?,?,?)";

        SQLiteStatement statement = database.compileStatement(sql);
        statement.clearBindings();

        statement.bindString(1, name);
        statement.bindLong(2,t);

        statement.bindLong(3,w);
        statement.bindLong(4,h);
        statement.bindBlob(5,bytes);
        statement.executeInsert();
    }

    public Mat dbgetcolor(String name) {
        SQLiteDatabase db = this.getReadableDatabase();
        String [] columns = {"t","w","h","pix"};
        Cursor cursor = this.getData("SELECT * FROM COLORF");

        if (cursor != null)
            cursor.moveToFirst();

        int t = cursor.getInt(2);
        int w = cursor.getInt(3);
        int h = cursor.getInt(4);
        byte[] p = cursor.getBlob(5);
        Mat m = new Mat(h,w,t);
        m.put(0,0,p);
        // Log.d("dbget("+name+")", m.toString());
        return m;
    }


    @Override
    public void onCreate(SQLiteDatabase sqLiteDatabase) {

    }

    @Override
    public void onUpgrade(SQLiteDatabase sqLiteDatabase, int i, int i1) {

    }
}
