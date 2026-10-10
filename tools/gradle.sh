#!/bin/sh
# Runs Unity's bundled Gradle 9.1 with Unity's JDK 17 in the android-probe project.
U="C:/Program Files/Unity 6000.3.24f1/Editor/Data/PlaybackEngines/AndroidPlayer"
export JAVA_HOME="$U/OpenJDK"
cd "$(dirname "$0")/../android-probe" || exit 1
exec "$JAVA_HOME/bin/java.exe" -Xmx64m -Dorg.gradle.appname=gradle -classpath "$U/Tools/gradle/lib/gradle-gradle-cli-main-9.1.0.jar" org.gradle.launcher.GradleMain --no-daemon "$@"
