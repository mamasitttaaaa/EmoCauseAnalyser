import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'db_helper.dart';
import 'dart:async';

// Точка входа в приложение
void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const MyApp());
}

// Константы для имени файла и адреса сервера
const audioFileName = "recorded.wav";
const serverURL = "https://1659-193-29-139-132.ngrok-free.app/";

// Флаг статуса функции работы с серевром и очередь для управления загрузками
bool _isUploading = false;
List<Completer<void>> _pendingUploadsQueue = [];

// Функция опроса сервера для получения результата обработки
Future<String?> pollForResult(String taskId) async {
  final statusUri = Uri.parse('${serverURL}check_status/$taskId');
  final resultUri = Uri.parse('${serverURL}result/$taskId');

  for (;;) {
    await Future.delayed(const Duration(seconds: 20));

    final response = await http.get(statusUri);
    final statusJson = jsonDecode(response.body);

    if (statusJson['status'] == 'done') {
      final resultResponse = await http.get(resultUri);
      final resultWrapper = jsonDecode(resultResponse.body);

      final bodyRaw = resultWrapper['result']?['body'];
      if (bodyRaw == null) throw Exception("Нет поля 'body' в ответе сервера");

      final body = jsonDecode(bodyRaw);
      final gptAnalysis = body['gpt_analysis'];
      if (gptAnalysis == null) return null;

      final decoded = utf8.decode(gptAnalysis.toString().runes.toList());
      if (decoded.trim().isEmpty) return null;

      return decoded.trim();
    } else if (statusJson['status'] == 'failed') {
      throw Exception("Обработка завершилась с ошибкой");
    }
  }
}

// Стартовая страница приложения
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // Проверка наличия сохранённого эталонного голоса
  Future<bool> hasSavedVoice() async {
    final path = await DBHelper.getOwnerPath();
    if (path != null) return File(path).existsSync();
    return false;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Emotion Tracker',
      debugShowCheckedModeBanner: false,
      home: StartPage(),
    );
  }
}

// Стартовая страница приложения
class StartPage extends StatelessWidget {
  const StartPage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Text("Добро пожаловать в", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            const Text("Emotion Cause Tracker", style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold)),
            const SizedBox(height: 30),
            ElevatedButton(
              child: const Text("Войти"),
              onPressed: () async {
                final hasEmbedding = await DBHelper.getOwnerPath().then((path) => path != null && File(path).existsSync());
                if (!context.mounted) return;
                if (hasEmbedding) {
                  Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const RecorderPage()));
                } else {
                  Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const ReferenceRecordingPage()));
                }
              },
            ),
          ],
        ),
      ),
    );
  }
}

// Страница записи эталонного голоса
class ReferenceRecordingPage extends StatefulWidget {
  final bool replace;
  const ReferenceRecordingPage({super.key, this.replace = false});
  @override
  State<ReferenceRecordingPage> createState() => _ReferenceRecordingPageState();
}

// Состояние страницы записи эталона
class _ReferenceRecordingPageState extends State<ReferenceRecordingPage> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;

  // Получение пути к файлу эталона
  Future<String> _getEmbeddingPath() async {
    final dir = await getApplicationDocumentsDirectory();
    return "${dir.path}/embedding.wav";
  }

  // Начало записи эталонного файла
  Future<void> _recordEmbedding() async {
    final path = await _getEmbeddingPath();
    await _recorder.openRecorder();
    await _recorder.startRecorder(toFile: path, codec: Codec.pcm16WAV);
    setState(() => _isRecording = true);
  }

  // Завершение записи эталонного файла
  Future<void> _stopEmbeddingRecording() async {
    await _recorder.stopRecorder();
    await _recorder.closeRecorder();
    setState(() => _isRecording = false);

    final path = await _getEmbeddingPath();
    await DBHelper.saveOwnerPath(path);

    if (!mounted) return;
    widget.replace ? Navigator.pop(context) : Navigator.pushReplacement(context, MaterialPageRoute(builder: (_) => const RecorderPage()));
  }

  // Отмена записи эталона и удаление файла
  Future<void> _cancelRecording() async {
    try {
      await _recorder.stopRecorder();
    } catch (_) {}
    await _recorder.closeRecorder();
    setState(() => _isRecording = false);

    final file = File(await _getEmbeddingPath());
    if (await file.exists()) await file.delete();
  }

  @override
  void dispose() {
    _recorder.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Запись эталона")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            const Text("Скажите фразу...", style: TextStyle(fontSize: 16)),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _isRecording ? _stopEmbeddingRecording : _recordEmbedding,
              child: Text(_isRecording ? "⏹️ Стоп" : "🎙️ Записать"),
            ),
            if (_isRecording)
              ElevatedButton(
                onPressed: _cancelRecording,
                child: const Text("❌ Отменить"),
              ),
          ],
        ),
      ),
    );
  }
}

// Страница записи и анализа пользовательского аудио
class RecorderPage extends StatefulWidget {
  const RecorderPage({super.key});
  @override
  State<RecorderPage> createState() => _RecorderPageState();
}

// Состояние страницы записи и анализа пользовательского аудио
class _RecorderPageState extends State<RecorderPage> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  bool _isRecording = false;
  List<Map<String, dynamic>> analysisHistory = [];
  DateTime? _recordStartTime;
  String? _recordedFilePath;

  @override
  void initState() {
    super.initState();
    _loadHistory(); // Загрузка истории анализов
    tryUploadPending(); // Запуск отложенной отправки
  }

  // Загрузка истории и сортировка по времени
  void _loadHistory() async {
    final history = List<Map<String, dynamic>>.from(await DBHelper.getHistory());

    // Сортировка от новых к старым
    history.sort((a, b) {
      final dateA = _parseDate(a['timestamp']);
      final dateB = _parseDate(b['timestamp']);
      return dateB.compareTo(dateA);
    });

    setState(() {
      analysisHistory = history;
    });
  }

  String getUniqueFileName() {
    final now = DateTime.now();
    return "recorded_${now.millisecondsSinceEpoch}.wav";
  }

  // Генерация уникального имени файла по текущему времени
  Future<String> _getFilePath() async {
    final dir = await getApplicationDocumentsDirectory();
    final fileName = getUniqueFileName();
    return "${dir.path}/$fileName";
  }

  // Получение пути к новому файлу записи
  Future<void> _startRecording() async {
    final path = await _getFilePath();
    _recordedFilePath = path;
    try {
      _recordStartTime = DateTime.now();
      await _recorder.openRecorder();
      await _recorder.startRecorder(toFile: path, codec: Codec.pcm16WAV);
      setState(() => _isRecording = true);
    } catch (e) {
      return;
    }
  }

  // Начало записи пользовательского аудио
  Future<void> _stopRecording() async {
    try {
      await _recorder.stopRecorder();
      if (_recorder.isStopped) await _recorder.closeRecorder();
      setState(() => _isRecording = false);
      await Future.delayed(const Duration(seconds: 1));
      final path = _recordedFilePath;
      if (path == null) {
        return;
      }
      final timestamp = _recordStartTime != null
        ? "${_recordStartTime!.day.toString().padLeft(2, '0')}.${_recordStartTime!.month.toString().padLeft(2, '0')}.${_recordStartTime!.year} "
          "${_recordStartTime!.hour.toString().padLeft(2, '0')}:${_recordStartTime!.minute.toString().padLeft(2, '0')}:${_recordStartTime!.second.toString().padLeft(2, '0')}"
        : "Время неизвестно";
      await DBHelper.addPendingUpload(path, timestamp);
      await tryUploadPending();
    } catch (e) {
      return;
    }
  }
  // Завершение записи и сохранение результата в историю
  Future<void> _cancelRecording() async {
    try {
      await _recorder.stopRecorder();
    } catch (_) {}
    await _recorder.closeRecorder();
    setState(() => _isRecording = false);

    final path = _recordedFilePath;
    if (path != null && File(path).existsSync()) {
      await File(path).delete();
    }
  }

  // Отправка аудио для обработки на сервер согласно очереди при доступности сервера
  Future<void> tryUploadPending() async {
    // Если уже идёт загрузка — создаём Completer и ждём
    if (_isUploading) {
      final completer = Completer<void>();
      _pendingUploadsQueue.add(completer);
      await completer.future;
      return; // Повторный вызов будет инициирован позже
    }

    _isUploading = true;

    try {
      final pending = await DBHelper.getPendingUploads();
      for (final item in pending) {
        final path = item['path'];
        final timestamp = item['timestamp'];
        final storedTaskId = item['task_id'];

        try {
          final file = File(path);
          if (!file.existsSync()){
            await DBHelper.removePendingUpload(path);
            continue;
          }

          final serverIsUp = await isServerAvailable();
          if (!serverIsUp) {
            break;
          }

          final embeddingPath = await DBHelper.getOwnerPath();
          if (embeddingPath == null || !File(embeddingPath).existsSync()) {
            continue;
          }

          String? taskId = storedTaskId;
          if (taskId == null || taskId.isEmpty) {
            final uri = Uri.parse('${serverURL}submit_audio');
            final request = http.MultipartRequest('POST', uri)
              ..files.add(await http.MultipartFile.fromPath('file', path))
              ..files.add(await http.MultipartFile.fromPath('owner_file', embeddingPath));

            final response = await request.send();
            final responseBody = await http.Response.fromStream(response);
            final json = jsonDecode(responseBody.body);
            taskId = json['task_id'];

            if (taskId == null || taskId.isEmpty) {
              continue;
            }

            await DBHelper.updateTaskIdForPending(path, taskId);
          }

          final result = await pollForResult(taskId);

          await DBHelper.removePendingUpload(path);
          try {
            final file = File(path);
            if (await file.exists()) {
              await file.delete();
            }
          } catch (e) {
            return;
          }

          if (result == null || result.isEmpty) {
            return;
          }

          final line = result.trim();
          await DBHelper.insertHistory(timestamp, line);

          _loadHistory();

        } catch (e) {
          return;
        }
      }

    } catch (e) {
      return;
    } finally {
      _isUploading = false;
      if (_pendingUploadsQueue.isNotEmpty) {
        final next = _pendingUploadsQueue.removeAt(0);
        Future.microtask(() => tryUploadPending()); // Автоматический запуск следующей
        next.complete(); // Снятие блокировки
      }
    }
  }

  // Проверка доступности сервера
  Future<bool> isServerAvailable() async {
    try {
      final uri = Uri.parse('${serverURL}ping');
      final response = await http.get(uri).timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        return decoded['status'] == 'ok';
      }
      return false;
    } catch (e) {
      return false;
    }
  }

  DateTime _parseDate(String dateString) {
    try {
      final parts = dateString.split(' ');
      final dateParts = parts[0].split('.');
      final timeParts = parts[1].split(':');

      return DateTime(
        int.parse(dateParts[2]),
        int.parse(dateParts[1]),
        int.parse(dateParts[0]),
        int.parse(timeParts[0]),
        int.parse(timeParts[1]),
        int.parse(timeParts[2]),
      );
    } catch (_) {
      return DateTime.fromMillisecondsSinceEpoch(0);
    }
  }

  @override
  void dispose() {
    if (_recorder.isStopped) _recorder.closeRecorder();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Emotion Tracker')),
      body: Column(
        children: [
          const SizedBox(height: 10),
          _isRecording
            ? Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Column(
                    children: [
                      ElevatedButton(
                        onPressed: _stopRecording,
                        child: const Text('⏹️ Стоп'),
                      ),
                    ],
                  ),
                  const SizedBox(width: 16),
                  Column(
                    children: [
                      ElevatedButton(
                        onPressed: _cancelRecording,
                        child: const Text('❌ Отменить'),
                      ),
                    ],
                  ),
                ],
              )
            : ElevatedButton(
                onPressed: _startRecording,
                child: const Text('🎙️ Запись'),
              ),
          ElevatedButton(
            onPressed: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const ReferenceRecordingPage(replace: true)),
              );
            },
            child: const Text("🔁 Заменить голос владельца"),
          ),
          const Divider(),
          const Text("История состояний:", style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(
            child: ListView.builder(
              itemCount: analysisHistory.length,
              itemBuilder: (context, index) {
                final entry = analysisHistory[index];
                final date = entry['timestamp'];
                final text = entry['result'];
                return ListTile(
                  title: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(date, style: const TextStyle(fontSize: 14, color: Colors.grey)),
                      Text(text, style: const TextStyle(fontSize: 16, color: Colors.black)),
                    ],
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
