import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import './output.dart';
import 'dart:io';
import 'package:tts_azure/tts_azure.dart';

class ImageInput extends StatefulWidget {
  static const routeName = '/ImageInput';

  @override
  _ImageInputState createState() => _ImageInputState();
}

class _ImageInputState extends State<ImageInput> with WidgetsBindingObserver {
  CameraController _controller;
  Future<void> _initcontroller;
  var isCameraReady = false;
  XFile imagefile;
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
    initCamera();
    WidgetsBinding.instance.addObserver(this);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (state == AppLifecycleState.resumed)
      _initcontroller = _controller != null ? _controller.initialize() : null;
    if (!mounted) return;
    setState(() {
      isCameraReady = true;
    });
  }

  Widget CameraWidget(context) {
    var camera = _controller.value;
    final size = MediaQuery.of(context).size;
    var scale = size.aspectRatio * camera.aspectRatio;
    if (scale < 1) scale = 1 / scale;
    return Transform.scale(
      alignment: Alignment.center,
      scale: scale,
      child: CameraPreview(_controller),
    );
  }

  String text = 'اضغط على الشاشة لاتقاط الصورة';
  var firsttime = 0;
  String lang = "ar-EG";
  String shortName = "ar-EG-Hoda"; // The voice.
  final ttsazure = TTSAzure("dba027714f7148f081fa56f326762abf", "eastus");

  @override
  Widget build(BuildContext context) {
    if (firsttime == 0) {
      Future.delayed(
          Duration.zero, () => ttsazure.speak(text, lang, shortName));
    }
    return Scaffold(
        body: FutureBuilder(
      future: _initcontroller,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          return Center(
              child: GestureDetector(
            child: CameraWidget(context),
            onTap: () => getimage(context),
          ));
        } else {
          return Center(
            child: CircularProgressIndicator(),
          );
        }
      },
    ));
  }

  Future<void> initCamera() async {
    final camera = await availableCameras();
    final firstCamera = camera.first;
    _controller = CameraController(firstCamera, ResolutionPreset.high);
    _initcontroller = _controller.initialize();
    if (!mounted) return;
    setState(() {
      isCameraReady = true;
    });
  }

  getimage(BuildContext context) {
    _controller.takePicture().then((file) {
      setState(() {
        firsttime = firsttime + 1;
        imagefile = file;
      });
      if (mounted) {
        Navigator.of(context)
            .pushNamed(
              VoiceHome.routeName,
              arguments: File(imagefile.path),
            )
            .then((value) => Future.delayed(
                Duration.zero, () => ttsazure.speak(text, lang, shortName)));
      }
    });
  }
}
