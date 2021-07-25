import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';
import './image_input.dart';
import './output.dart';

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
  final FlutterTts flutterTts = FlutterTts();
  Future speak(text) async {
    await flutterTts.setPitch(0.8);
    await flutterTts.setLanguage("en-US");
    await flutterTts.speak(text);
  }

  String text = 'Welcome to shoofli, tap to screen to start the camera';

  @override
  Widget build(BuildContext context) {
    Future.delayed(Duration.zero, () => speak(text));
    return Scaffold(
        body: GestureDetector(
            child: Center(
              child: Text(
                'Welcome to shoofli tap to screen to start the camera',
              ),
            ),
            onTap: () {
              Navigator.of(context)
                  .pushNamed(
                    ImageInput.routeName,
                  )
                  .then((value) =>
                      Future.delayed(Duration.zero, () => speak(text)));
            }));
  }
}
