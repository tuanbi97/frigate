import { Fragment, h } from 'preact';
import CameraBBox from './CameraBBox';
import { useCallback, useEffect, useState } from 'preact/hooks';

const MIN_LOAD_TIMEOUT_MS = 200;

export default function AutoUpdatingCameraBBox({ camera, searchParams = '', showFps = true }) {
  const [key, setKey] = useState(Date.now());
  const [fps, setFps] = useState(0);

  const handleLoad = useCallback(() => {
    const loadTime = Date.now() - key;
    // console.log(loadTime);
    setFps((1000 / Math.max(loadTime, MIN_LOAD_TIMEOUT_MS)).toFixed(1));
    setTimeout(
      () => {
        setKey(Date.now());
      },
      // loadTime > MIN_LOAD_TIMEOUT_MS ? 1 : MIN_LOAD_TIMEOUT_MS
      1
    );
  }, [key, setFps]);

  return (
    <Fragment>
      <CameraBBox camera={camera} onload={handleLoad} searchParams={`cache=${key}&${searchParams}`} />
      {/* {showFps ? <span className="text-xs">Displaying at {fps}fps</span> : null} */}
    </Fragment>
  );
}