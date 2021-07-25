import 'package:flutter/material.dart';
import './image_input.dart';
import './output.dart';
import 'package:tts_azure/tts_azure.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Shoofli',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: MyHomePage(title: 'shoofli App'),
      initialRoute: '/', // default is '/'
      routes: {
        ImageInput.routeName: (ctx) => ImageInput(),
        VoiceHome.routeName: (ctx) => VoiceHome(),
      },
    );
  }
}

class MyHomePage extends StatefulWidget {
  MyHomePage({Key key, this.title}) : super(key: key);

  final String title;

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  String lang = "ar-EG";
  String shortName = "ar-EG-Hoda"; // The voice.
  final ttsazure = TTSAzure("dba027714f7148f081fa56f326762abf", "eastus");

  String text = 'اهلا فى شوفلي اضغط على الشاشة لفتح الكاميرا';

  @override
  Widget build(BuildContext context) {
    Future.delayed(Duration.zero, () => ttsazure.speak(text, lang, shortName));
    return Scaffold(
        body: GestureDetector(
            child: Center(
              child: Text(
                'اهلا فى شوفلي اضغط على الشاشة لفتح الكاميرا',
              ),
            ),
            onTap: () {
              Navigator.of(context)
                  .pushNamed(
                    ImageInput.routeName,
                  )
                  .then((value) => Future.delayed(Duration.zero,
                      () => ttsazure.speak(text, lang, shortName)));
            }));
  }
}
