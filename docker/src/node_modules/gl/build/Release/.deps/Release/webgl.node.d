cmd_Release/webgl.node := ln -f "Release/obj.target/webgl.node" "Release/webgl.node" 2>/dev/null || (rm -rf "Release/webgl.node" && cp -af "Release/obj.target/webgl.node" "Release/webgl.node")
