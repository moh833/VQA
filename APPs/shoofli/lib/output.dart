import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:io';
import 'package:flutter_tts/flutter_tts.dart';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'package:google_speech/google_speech.dart';
import 'package:google_speech/speech_to_text_beta.dart';
import 'package:path_provider/path_provider.dart';
import 'package:rxdart/rxdart.dart';
import 'package:sound_stream/sound_stream.dart';
import 'dart:async';

class VoiceHome extends StatefulWidget {
  static const routeName = '/Voiceout';
  @override
  _VoiceHomeState createState() => _VoiceHomeState();
}

class _VoiceHomeState extends State<VoiceHome> {
  int firsttime = 0;
  bool isListening = false;
  File imgs;
  String mode;
  String ans;
  String prevQuestion;
  final RecorderStream _recorder = RecorderStream();
  StreamSubscription<List<int>> _audioStreamSubscription;
  BehaviorSubject<List<int>> _audioStream;

  bool recognizing = false;
  bool recognizeFinished = false;
  String text = '';
  void initState() {
    super.initState();

    _recorder.initialize();
  }

  void streamingRecognize() async {
    _audioStream = BehaviorSubject<List<int>>();
    _audioStreamSubscription = _recorder.audioStream.listen((event) {
      _audioStream.add(event);
    });

    await _recorder.start();

    setState(() {
      recognizing = true;
    });
    final serviceAccount = ServiceAccount.fromString(
        '${(await rootBundle.loadString('assets/test_service_account.json'))}');
    final speechToText = SpeechToText.viaServiceAccount(serviceAccount);
    final config = _getConfig();
    final responseStream = speechToText.streamingRecognize(
        StreamingRecognitionConfig(config: config, interimResults: true),
        _audioStream);

    var responseText = '';

    responseStream.listen((data) {
      final currentText =
          data.results.map((e) => e.alternatives.first.transcript).join('\n');

      if (data.results.first.isFinal) {
        responseText += currentText;
        setState(() {
          text = responseText;
          recognizeFinished = true;
        });
      } else {
        setState(() {
          text = responseText + currentText;
          recognizeFinished = true;
        });
      }
    }, onDone: () {
      setState(() {
        recognizing = false;
      });
    });
  }

  void stopRecording(text) async {
    await _recorder.stop();
    print(text);
    await _audioStreamSubscription?.cancel();
    await _audioStream?.close();
    Future.delayed(Duration(seconds: 1), () async {
      say('your question is  $text, double click to confirm');
    });
    setState(() {
      recognizing = false;
    });
  }

  RecognitionConfig _getConfig() => RecognitionConfig(
      encoding: AudioEncoding.LINEAR16,
      model: RecognitionModel.basic,
      enableAutomaticPunctuation: true,
      sampleRateHertz: 16000,
      languageCode: 'en-US');

  Widget build(BuildContext context) {
    File img = ModalRoute.of(context).settings.arguments;
    imgs = File(img.path);
    if (firsttime == 0) {
      Future.delayed(
          Duration.zero,
          () => _speak(
              'Tap to screen to ask your question, then tap again to confirm'));
      firsttime = firsttime + 1;
    }

    return Scaffold(
        body: GestureDetector(
      onTap: recognizing
          ? () {
              stopRecording(text);
            }
          : streamingRecognize,
      onDoubleTap: () {
        getResult();
      },
      child: Center(
        child: Image.file(
          imgs,
          fit: BoxFit.cover,
          width: double.infinity,
        ),
      ),
    ));
  }

  final FlutterTts flutterTts = FlutterTts();
  Future _speak(text) async {
    await flutterTts.setPitch(0.75);
    await flutterTts.setSpeechRate(0.4);
    await flutterTts.setLanguage("en-US");
    await flutterTts.speak(text);
  }

  Future say(String typ) async {
    Future.delayed(Duration.zero, () => _speak(typ));
  }

  Future getResult() {
    if (prevQuestion == text) {
      say('the answer is:  $ans');
    } else {
      prevQuestion = text;
      Future.delayed(Duration(seconds: 0), () async {
        final urs = 'http://20.85.240.203:5000/';
        Uri erl = Uri.parse(urs) as Uri;
        String base64Image = base64Encode(imgs.readAsBytesSync());
        print(text);
        print(mode);
        final response = await http.post(
          Uri.parse(urs),
          body: jsonEncode(
            {'image': base64Image, 'question': text},
          ),
          headers: {'Content-Type': "application/json"},
        );
        ans = response.body;
        say('the answer is:  ${response.body}');
      });
    }
  }
}
