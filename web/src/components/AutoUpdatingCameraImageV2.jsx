import { h } from 'preact';
import CameraImageV2 from './CameraImageV2';
import useSWR from 'swr';
import { useCallback, useState, useEffect } from 'preact/hooks';

export default function AutoUpdatingCameraImageV2({ camera, searchParams = '', showFps = true, className }) {
  const { data: config } = useSWR('config');
  const [key, setKey] = useState(Date.now());
  const [fps, setFps] = useState(0);
  const cameraFps = config ? config.cameras[camera].detect.fps : 5;
  const minLoadTimeoutMs = 1000 / cameraFps;

  const handleLoad = useCallback((status) => {
    const loadTime = Date.now() - key;
    setFps((1000 / Math.max(loadTime, minLoadTimeoutMs)).toFixed(1));
    setTimeout(
      () => {
        // if (status == "loaded")
        //   setKey(key + minLoadTimeoutMs);
        // else {
        //   console.log("waiting");
        //   setKey(key + 1);
        // }
        setKey(Date.now())
      },
      // minLoadTimeoutMs
      loadTime > minLoadTimeoutMs ? 1 : minLoadTimeoutMs
    );
  }, [key, setFps]);

  // const handleLoad = null;
  // useEffect(() => {
  //   const loadTime = Date.now() - key;
  //   setFps((1000 / Math.max(loadTime, minLoadTimeoutMs)).toFixed(1));
  //   setTimeout(
  //     () => {
  //       setKey(Date.now());
  //     },
  //     loadTime > minLoadTimeoutMs ? 1 : minLoadTimeoutMs - loadTime
  //   );
  // }, [key, setFps])

  return (
    <div className={className}>
      <CameraImageV2 camera={camera} searchParams={`cache=${key}&${searchParams}`} onload={handleLoad} frameInterval={minLoadTimeoutMs} />
      {showFps ? <span className="text-xs">Displaying at {fps}fps</span> : null}
    </div>
  );
}
