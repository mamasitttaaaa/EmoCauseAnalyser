import 'dart:io';
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sqflite/sqflite.dart';

class DBHelper {
  static Database? _db;

  static Future<Database> get db async {
    if (_db != null) return _db!;
    _db = await _initDB();
    return _db!;
  }

  static Future<Database> _initDB() async {
    Directory dir = await getApplicationDocumentsDirectory();
    final path = join(dir.path, 'emotion_tracker.db');
    return await openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('CREATE TABLE owner_voice(id INTEGER PRIMARY KEY, path TEXT)');
        await db.execute('CREATE TABLE history(id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, result TEXT)');
        await db.execute('CREATE TABLE pending_uploads(id INTEGER PRIMARY KEY AUTOINCREMENT, path TEXT, timestamp TEXT, task_id TEXT)');
      },
    );
  }

  static Future<void> addPendingUpload(String path, String timestamp, {String? taskId}) async {
    final db = await DBHelper.db;
    await db.insert('pending_uploads', {
      'path': path,
      'timestamp': timestamp,
      'task_id': taskId,
    });
  }

  static Future<void> updateTaskIdForPending(String path, String taskId) async {
    final db = await DBHelper.db;
    await db.update(
      'pending_uploads',
      {'task_id': taskId},
      where: 'path = ?',
      whereArgs: [path],
    );
  }

  static Future<void> saveOwnerPath(String path) async {
    final db = await DBHelper.db;
    await db.delete('owner_voice');
    await db.insert('owner_voice', {'id': 1, 'path': path});
  }

  static Future<String?> getOwnerPath() async {
    final db = await DBHelper.db;
    final result = await db.query('owner_voice', where: 'id = ?', whereArgs: [1]);
    return result.isNotEmpty ? result.first['path'] as String : null;
  }

  static Future<void> insertHistory(String timestamp, String result) async {
    final db = await DBHelper.db;
    await db.insert('history', {'timestamp': timestamp, 'result': result});
  }

  static Future<List<Map<String, dynamic>>> getHistory() async {
    final db = await DBHelper.db;
    return await db.query('history', orderBy: 'id DESC');
  }

  static Future<void> clearAll() async {
    final db = await DBHelper.db;
    await db.delete('owner_voice');
    await db.delete('history');
  }

  static Future<List<Map<String, dynamic>>> getPendingUploads() async {
    final db = await DBHelper.db;
    return await db.query('pending_uploads');
  }

  static Future<void> removePendingUpload(String path) async {
    final db = await DBHelper.db;
    await db.delete('pending_uploads', where: 'path = ?', whereArgs: [path]);
  }

  static Future<void> clearAllFinal() async {
    final db = await DBHelper.db;
    await db.delete('history');
    await db.delete('pending_uploads');
    await db.delete('owner_voice');
  }
}