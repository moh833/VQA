import 'package:flutter/material.dart';
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:flutter/services.dart';
import 'package:google_speech/google_speech.dart';
import 'package:google_speech/speech_to_text_beta.dart';
import 'package:path_provider/path_provider.dart';
import 'package:rxdart/rxdart.dart';
import 'package:sound_stream/sound_stream.dart';
import 'dart:async';
import 'package:translator/translator.dart';
import 'package:tts_azure/tts_azure.dart';

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
  final translator = GoogleTranslator();
  String lang = "ar-EG";
  String shortName = "ar-EG-Hoda"; // The voice.
  final ttsazure = TTSAzure("dba027714f7148f081fa56f326762abf", "eastus");

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
    }, onDone: () async {
      //await ttsazure.speak(" السؤال هو ", lang, shortName);
      var x = " السؤال هو ";
      var y = " اضغط على الشاشة مرتين للتأكيد";
      String txt = x + text + y;
      await ttsazure.speak(txt, lang, shortName);
      //await ttsazure.speak(" اضغط على الشاشة مرتين للتأكيد", lang, shortName);
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

    setState(() {
      recognizing = false;
    });
  }

  RecognitionConfig _getConfig() => RecognitionConfig(
      encoding: AudioEncoding.LINEAR16,
      model: RecognitionModel.basic,
      enableAutomaticPunctuation: true,
      sampleRateHertz: 16000,
      languageCode: 'ar-EG');

  Widget build(BuildContext context) {
    File img = ModalRoute.of(context).settings.arguments;
    imgs = File(img.path);
    if (firsttime == 0) {
      Future.delayed(
          Duration.zero,
          () => ttsazure.speak(
              'اضغط على الشاشة واسأل سؤالك ثم اضغط مرة اخري عند الانتهاء',
              lang,
              shortName));
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

  Future getResult() {
    if (prevQuestion == text) {
      ttsazure.speak("الاجابة هي $ans  ", lang, shortName);
    } else {
      prevQuestion = text;
      Future.delayed(Duration(seconds: 0), () async {
        final urs = 'http://20.85.240.203:5000/';
        Uri erl = Uri.parse(urs) as Uri;
        String base64Image = base64Encode(imgs.readAsBytesSync());
        print(text);
        print(mode);
        await translator
            .translate(text, from: 'auto', to: 'en')
            .then((value) => text = "$value");
        print(text);

        prevQuestion = text;

        final response = await http.post(
          Uri.parse(urs),
          body: jsonEncode(
            {'image': base64Image, 'question': text},
          ),
          headers: {'Content-Type': "application/json"},
        );
        ans = response.body;
        translator.translate(ans, from: 'en', to: 'ar').then((value) {
          setState(() {
            ans = "$value";
            ttsazure.speak("الاجابة هي  $value  ", lang, shortName);
          });
          ;
          print("$value");
        });
        //ttsazure.speak(" $ans الاجابة هي ", lang, shortName);
      });
    }
  }
}
